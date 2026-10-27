use anyhow::Result;
use femtovg::FontId;
use relm4::{
    Sender,
    gtk::{
        self,
        gdk::{Key, ModifierType},
        prelude::WidgetExt,
    },
};
use std::cell::Cell;

use crate::{
    configuration::APP_CONFIG,
    math::{
        SegmentResizeTarget, Vec2D, ensure_bounding_box, get_closest_aspect_ratio,
        homography::{Homography, Quad, outset_quad},
    },
    sketch_board::{KeyEventMsg, MouseButton, MouseEventMsg, MouseEventType, SketchBoardInput},
    tools::{
        Projected, RenderingMode,
        drag_box::{draw_center_marker, draw_quad_marker, draw_rect_marker},
    },
};

use super::{Drawable, InputContext, Tool, ToolUpdateResult, Tools};

// Desired on-screen size (in device pixels) for each resize handle.
const HANDLE_SIZE: f32 = 11.0;
const HANDLE_HALF: f32 = HANDLE_SIZE / 2.0;
const SELECTION_BORDER_OUTSET: f32 = 4.0;
// halvings of the drag that look for where a distorted quad stops being convex
const DISTORT_SEARCH_STEPS: usize = 16;

// Ctrl and Shift each already change a resize on their own
fn is_distort(modifier: ModifierType) -> bool {
    modifier.contains(ModifierType::CONTROL_MASK | ModifierType::SHIFT_MASK)
}

/// The selection frame: the bounds of the selected drawable in its own plane,
/// and the projection of that plane onto the image, if it has one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    tl: Vec2D,
    br: Vec2D,
    h: Option<Homography>,
}

impl Frame {
    pub fn of(drawable: &dyn Drawable) -> Option<Self> {
        match drawable.perspective() {
            Some((h, (tl, br))) => Some(Self { tl, br, h: Some(h) }),
            None => drawable.bounds().map(|(tl, br)| {
                let (tl, br) = ensure_bounding_box(tl, br);
                Self { tl, br, h: None }
            }),
        }
    }

    // `p` in the plane, where it is on the image
    fn on_image(&self, p: Vec2D) -> Vec2D {
        self.h.map_or(p, |h| h.apply(p))
    }

    fn quad(&self) -> Quad {
        let (tl, br) = (self.tl, self.br);
        [tl, Vec2D::new(br.x, tl.y), br, Vec2D::new(tl.x, br.y)].map(|p| self.on_image(p))
    }

    fn handle_pos(&self, handle: ResizeHandle) -> Vec2D {
        self.on_image(handle.center(self.tl, self.br))
    }

    fn with_bounds(&self, tl: Vec2D, br: Vec2D) -> Self {
        let (tl, br) = ensure_bounding_box(tl, br);
        Self { tl, br, h: self.h }
    }

    fn translated(&self, delta: Vec2D) -> Self {
        match self.h {
            Some(h) => Self {
                h: Some(h.then_translate(delta)),
                ..*self
            },
            None => self.with_bounds(self.tl + delta, self.br + delta),
        }
    }

    // the drag from `start` by `delta` on the image, as a drag in the plane
    fn plane_delta(&self, start: Vec2D, delta: Vec2D) -> Vec2D {
        match self.h.and_then(|h| h.inverse()) {
            // past the horizon of the plane, the pointer has no place in it
            Some(inverse) if inverse.denominator(start + delta) <= 0.0 => Vec2D::zero(),
            Some(inverse) => inverse.apply(start + delta) - inverse.apply(start),
            None => delta,
        }
    }

    // `drawable`, whose frame this is, projected so that the corners under
    // `handle` move by `delta` on the image; a corner stops where the
    // projection would no longer be accepted
    fn distort(
        &self,
        drawable: &dyn Drawable,
        handle: ResizeHandle,
        delta: Vec2D,
    ) -> Option<Box<dyn Drawable>> {
        let quad = self.quad();
        let moved = |t: f32| {
            let mut q = quad;
            for &i in handle.corners() {
                q[i] += delta * t;
            }
            q
        };
        let accepted = |t: f32| Projected::projection(self.tl, self.br, &moved(t)).is_some();
        let fraction = if accepted(1.0) {
            1.0
        } else {
            let (mut valid, mut invalid) = (0.0, 1.0);
            for _ in 0..DISTORT_SEARCH_STEPS {
                let t = (valid + invalid) / 2.0;
                if accepted(t) {
                    valid = t;
                } else {
                    invalid = t;
                }
            }
            valid
        };
        Projected::project(drawable, &moved(fraction))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResizeHandle {
    TopLeft,
    TopCenter,
    TopRight,
    MiddleLeft,
    MiddleRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl ResizeHandle {
    pub fn all() -> [ResizeHandle; 8] {
        [
            ResizeHandle::TopLeft,
            ResizeHandle::TopCenter,
            ResizeHandle::TopRight,
            ResizeHandle::MiddleLeft,
            ResizeHandle::MiddleRight,
            ResizeHandle::BottomLeft,
            ResizeHandle::BottomCenter,
            ResizeHandle::BottomRight,
        ]
    }

    // the indices into a `Quad` of the corners a distortion moves
    fn corners(&self) -> &'static [usize] {
        match self {
            ResizeHandle::TopLeft => &[0],
            ResizeHandle::TopCenter => &[0, 1],
            ResizeHandle::TopRight => &[1],
            ResizeHandle::MiddleRight => &[1, 2],
            ResizeHandle::BottomRight => &[2],
            ResizeHandle::BottomCenter => &[2, 3],
            ResizeHandle::BottomLeft => &[3],
            ResizeHandle::MiddleLeft => &[3, 0],
        }
    }

    pub fn center(&self, tl: Vec2D, br: Vec2D) -> Vec2D {
        let mx = (tl.x + br.x) / 2.0;
        let my = (tl.y + br.y) / 2.0;
        match self {
            ResizeHandle::TopLeft => tl,
            ResizeHandle::TopCenter => Vec2D::new(mx, tl.y),
            ResizeHandle::TopRight => Vec2D::new(br.x, tl.y),
            ResizeHandle::MiddleLeft => Vec2D::new(tl.x, my),
            ResizeHandle::MiddleRight => Vec2D::new(br.x, my),
            ResizeHandle::BottomLeft => Vec2D::new(tl.x, br.y),
            ResizeHandle::BottomCenter => Vec2D::new(mx, br.y),
            ResizeHandle::BottomRight => br,
        }
    }

    fn segment_target(&self, tl: Vec2D, br: Vec2D) -> Option<SegmentResizeTarget> {
        if tl.y == br.y && tl.x != br.x {
            // horizontal segment
            match self {
                Self::TopLeft | Self::MiddleLeft | Self::BottomLeft => {
                    Some(SegmentResizeTarget::Low)
                }
                Self::TopRight | Self::MiddleRight | Self::BottomRight => {
                    Some(SegmentResizeTarget::High)
                }
                Self::TopCenter | Self::BottomCenter => Some(SegmentResizeTarget::MiddleY),
            }
        } else if tl.x == br.x && tl.y != br.y {
            // vertical segment
            match self {
                Self::TopLeft | Self::TopCenter | Self::TopRight => Some(SegmentResizeTarget::Low),
                Self::BottomLeft | Self::BottomCenter | Self::BottomRight => {
                    Some(SegmentResizeTarget::High)
                }
                Self::MiddleLeft | Self::MiddleRight => Some(SegmentResizeTarget::MiddleX),
            }
        } else {
            // diagonal segment: only the edge-center handles move the whole segment
            match self {
                Self::TopCenter | Self::BottomCenter => Some(SegmentResizeTarget::MiddleY),
                Self::MiddleLeft | Self::MiddleRight => Some(SegmentResizeTarget::MiddleX),
                _ => None,
            }
        }
    }

    // Compute new (tl, br) after dragging this handle by `delta`.
    //
    // This intentionally preserves axis inversion (tl may become > br) so tools like
    // line/arrow can keep endpoint intent when crossing over an axis.
    pub fn resize(&self, event: MouseEventMsg, tl: Vec2D, br: Vec2D) -> (Vec2D, Vec2D) {
        let mut new_tl = tl;
        let mut new_br = br;
        let mut delta = event.pos;

        let aspect = event.modifier.intersects(ModifierType::SHIFT_MASK);
        let keep_aspect = event.modifier.intersects(ModifierType::CONTROL_MASK);
        let centered = event.modifier.intersects(ModifierType::ALT_MASK);

        type RH = ResizeHandle;

        // keep aspect ratio if Ctrl is held down on corners
        if keep_aspect && (br.y - tl.y) > f32::EPSILON {
            let size = br - tl;
            let aspect_ratio = size.x.abs() / size.y.abs();
            let center_scale = if centered { 1.0 } else { 2.0 };

            // Adjusts a (width-delta, height-delta) pair, in the "both attached to
            // br" sign convention, so it maintains aspect_ratio. The axis with the
            // larger resulting extent is kept (so the pointer stays on that edge) and
            // each axis keeps the sign the pointer put it in.
            // When centered, wh is applied to *both* opposite corners, so the
            // real total size change is 2*wh, not wh.
            let total_scale = 2.0 / center_scale;
            let fix_aspect = |wh: Vec2D| -> Vec2D {
                let s = size + wh * total_scale;
                let sx = if s.x < 0.0 { -1.0 } else { 1.0 };
                let sy = if s.y < 0.0 { -1.0 } else { 1.0 };
                let target = if s.x.abs() >= s.y.abs() * aspect_ratio {
                    Vec2D::new(s.x, sy * s.x.abs() / aspect_ratio)
                } else {
                    Vec2D::new(sx * s.y.abs() * aspect_ratio, s.y)
                };
                (target - size) / total_scale
            };

            // Total size change along the other axis so that the magnitude follows
            // `new_extent` while keeping that axis' original sign.
            let follow_x = |new_h: f32| size.x.signum() * new_h.abs() * aspect_ratio - size.x;
            let follow_y = |new_w: f32| size.y.signum() * new_w.abs() / aspect_ratio - size.y;

            match self {
                RH::TopLeft => {
                    // w and h are both attached to tl (-dx/-dy grow them)
                    let wh = fix_aspect(delta * -1.0);
                    delta = wh * -1.0;
                }
                RH::BottomRight => {
                    // w and h are both attached to br (dx/dy grow them)
                    delta = fix_aspect(delta);
                }
                RH::TopRight => {
                    // w is attached to br (dx grows it), h to tl (-dy grows it).
                    let wh = fix_aspect(Vec2D::new(delta.x, -delta.y));
                    delta = Vec2D::new(wh.x, -wh.y);
                }
                RH::BottomLeft => {
                    // w is attached to tl (-dx grows it), h to br (dy grows it).
                    let wh = fix_aspect(Vec2D::new(-delta.x, delta.y));
                    delta = Vec2D::new(-wh.x, wh.y);
                }
                RH::TopCenter => {
                    // h is attached to tl (-dy grows it); w follows via aspect ratio.
                    delta.x = follow_x(size.y - delta.y * total_scale) / total_scale;
                    new_tl.x -= delta.x / center_scale;
                    new_br.x += delta.x / center_scale;
                }
                RH::BottomCenter => {
                    // h is attached to br (dy grows it); w follows via aspect ratio.
                    delta.x = follow_x(size.y + delta.y * total_scale) / total_scale;
                    new_tl.x -= delta.x / center_scale;
                    new_br.x += delta.x / center_scale;
                }
                RH::MiddleLeft => {
                    // w is attached to tl (-dx grows it); h follows via aspect ratio.
                    delta.y = follow_y(size.x - delta.x * total_scale) / total_scale;
                    new_tl.y -= delta.y / center_scale;
                    new_br.y += delta.y / center_scale;
                }
                RH::MiddleRight => {
                    // w is attached to br (dx grows it); h follows via aspect ratio.
                    delta.y = follow_y(size.x + delta.x * total_scale) / total_scale;
                    new_tl.y -= delta.y / center_scale;
                    new_br.y += delta.y / center_scale;
                }
            }
        }

        // keep center fixed if Alt is held down
        match self {
            RH::TopLeft => {
                new_tl += delta;
                if centered {
                    new_br -= delta;
                }
            }
            RH::BottomRight => {
                new_br += delta;
                if centered {
                    new_tl -= delta;
                }
            }
            RH::TopRight => {
                new_tl.y += delta.y;
                new_br.x += delta.x;
                if centered {
                    new_tl.x -= delta.x;
                    new_br.y -= delta.y;
                }
            }
            RH::BottomLeft => {
                new_tl.x += delta.x;
                new_br.y += delta.y;
                if centered {
                    new_tl.y -= delta.y;
                    new_br.x -= delta.x;
                }
            }
            RH::TopCenter => {
                new_tl.y += delta.y;
                if centered {
                    new_br.y -= delta.y;
                }
            }
            RH::BottomCenter => {
                new_br.y += delta.y;
                if centered {
                    new_tl.y -= delta.y;
                }
            }
            RH::MiddleLeft => {
                new_tl.x += delta.x;
                if centered {
                    new_br.x -= delta.x;
                }
            }
            RH::MiddleRight => {
                new_br.x += delta.x;
                if centered {
                    new_tl.x -= delta.x;
                }
            }
        }

        // keep to predefined aspect ratio if Shift is held down
        if aspect {
            let w = new_br.x - new_tl.x;
            let h = new_br.y - new_tl.y;
            let config = APP_CONFIG.read();

            // Ratio is magnitude only; signs are kept so an inverted axis stays inverted
            // and the dragged edge stays under the cursor.
            let closest_aspect_ratio =
                get_closest_aspect_ratio((w / h).abs(), config.aspect_ratios());
            let aspect_ratio = closest_aspect_ratio.0 / closest_aspect_ratio.1;
            let sx = if w < 0.0 { -1.0 } else { 1.0 };
            let sy = if h < 0.0 { -1.0 } else { 1.0 };
            let size = if h.abs() <= f32::EPSILON {
                Vec2D::new(w, h) // fallback
            } else {
                match self {
                    RH::TopCenter | RH::BottomCenter => Vec2D::new(sx * h.abs() * aspect_ratio, h),
                    RH::MiddleLeft | RH::MiddleRight => Vec2D::new(w, sy * w.abs() / aspect_ratio),
                    _ => {
                        if w.abs() >= h.abs() * aspect_ratio {
                            Vec2D::new(w, sy * w.abs() / aspect_ratio)
                        } else {
                            Vec2D::new(sx * h.abs() * aspect_ratio, h)
                        }
                    }
                }
            };

            let size_half = size / 2.0;
            let center = (new_tl + new_br) / 2.0;

            if centered {
                new_tl = center - size_half;
                new_br = center + size_half;
            } else {
                match self {
                    RH::TopLeft => {
                        new_tl = new_br - size;
                    }
                    RH::TopRight => {
                        new_br.x = new_tl.x + size.x;
                        new_tl.y = new_br.y - size.y;
                    }
                    RH::BottomRight => {
                        new_br = new_tl + size;
                    }
                    RH::BottomLeft => {
                        new_tl.x = new_br.x - size.x;
                        new_br.y = new_tl.y + size.y;
                    }
                    RH::TopCenter | RH::BottomCenter => {
                        new_tl.x = center.x - size_half.x;
                        new_br.x = center.x + size_half.x;
                    }
                    RH::MiddleLeft | RH::MiddleRight => {
                        new_tl.y = center.y - size_half.y;
                        new_br.y = center.y + size_half.y;
                    }
                }
            }
        }

        (new_tl, new_br)
    }
}

// Returns the handle under `pos`, if any, given the selection frame.
fn hit_handle(scaled_handle_size: f32, pos: Vec2D, frame: &Frame) -> Option<ResizeHandle> {
    for h in ResizeHandle::all() {
        let handle_half = scaled_handle_size / 2.0;

        let c = frame.handle_pos(h);
        if (pos.x - c.x).abs() <= handle_half && (pos.y - c.y).abs() <= handle_half {
            return Some(h);
        }
    }
    None
}

// Draws a selection rectangle with 8 resize handles on top of the selected drawable.
#[derive(Clone, Debug)]
struct SelectionOverlay {
    frame: Frame,
    scale: Cell<f32>,
    centered: bool,
}

impl Drawable for SelectionOverlay {
    fn get_rendering_mode(&self) -> RenderingMode {
        RenderingMode::SelectionOverlay
    }

    fn draw(
        &self,
        canvas: &mut femtovg::Canvas<femtovg::renderer::OpenGl>,
        _font: FontId,
        _bounds: (Vec2D, Vec2D),
    ) -> Result<()> {
        let scale = canvas.transform().average_scale().max(f32::EPSILON);
        // to be used at other places where the scale is not available
        self.scale.set(scale);

        // Selection rectangle, or the quad of a projected drawable; the bounds
        // of a straight line have no area, which only the rectangle can outset
        let out_set = SELECTION_BORDER_OUTSET / scale;
        if self.frame.h.is_some() {
            draw_quad_marker(canvas, &outset_quad(&self.frame.quad(), out_set), false);
        } else {
            let tl = self.frame.tl - out_set;
            let size = (self.frame.br - self.frame.tl) + out_set * 2.0;
            draw_rect_marker(canvas, tl, size, false);
        }

        // Resize handles
        // draw handles in inverse zoom scale so the visual size stays constant on screen.
        let handle_half = HANDLE_HALF / scale;
        let handle_size = HANDLE_SIZE / scale;
        for handle in ResizeHandle::all() {
            let tl = self.frame.handle_pos(handle) - handle_half;
            draw_rect_marker(canvas, tl, Vec2D::new(handle_size, handle_size), true);
        }

        if self.centered {
            let frame = &self.frame;
            draw_center_marker(canvas, frame.on_image((frame.tl + frame.br) / 2.0));
        }

        Ok(())
    }
}

#[derive(Debug)]
enum DragState {
    None,
    Moving {
        index: usize,
        original: Box<dyn Drawable>,
        orig_frame: Frame,
    },
    Resizing {
        index: usize,
        original: Box<dyn Drawable>,
        handle: ResizeHandle,
        orig_frame: Frame,
    },
}

pub struct PointerTool {
    input_enabled: bool,
    sender: Option<Sender<SketchBoardInput>>,
    cursor_widget: Option<gtk::Widget>,
    selected_index: Option<usize>,
    selected_frame: Option<Frame>,
    drag_state: DragState,
    last_drag_state: bool,
    // Shown as the active-tool drawable: either a moved/resized preview, or a selection overlay.
    preview: Option<Box<dyn Drawable>>,
    selection_overlay: Option<SelectionOverlay>,
    // For cycling through overlapping objects: last click position
    last_click_pos: Option<Vec2D>,
    // For cycling through overlapping objects: all hit objects at last click position
    hit_objects_at_pos: Vec<usize>,
    // For cycling through overlapping objects: current index in hit_objects list
    current_cycle_index: usize,
    // Absolute pointer position (image coordinates) at drag start.
    drag_start_pos: Option<Vec2D>,
    // Used for toggling centered mode
    last_resize_event: Option<MouseEventMsg>,
    // Whether the selection should display center marker.
    centered: bool,
}

impl Default for PointerTool {
    fn default() -> Self {
        Self {
            input_enabled: false,
            sender: None,
            cursor_widget: None,
            selected_index: None,
            selected_frame: None,
            drag_state: DragState::None,
            last_drag_state: false,
            preview: None,
            selection_overlay: None,
            last_click_pos: None,
            hit_objects_at_pos: Vec::new(),
            current_cycle_index: 0,
            drag_start_pos: None,
            last_resize_event: None,
            centered: false,
        }
    }
}

impl PointerTool {
    pub fn get_cursor(&self, name: &str) -> Option<gtk::gdk::Cursor> {
        let cursor_candidates = match name {
            "grabbing" => Some(&["grabbing", "all-resize"]),
            "grab" => Some(&["grab", "all-scroll"]),
            "nwse-resize" => Some(&["nwse-resize", "top-left-corner"]),
            "nesw-resize" => Some(&["nesw-resize", "top-right-corner"]),
            "ns-resize" => Some(&["ns-resize", "top-center"]),
            "ew-resize" => Some(&["ew-resize", "middle-left"]),
            "not-allowed" => Some(&["not-allowed", "no-drop"]),
            _ => None,
        };
        cursor_candidates.and_then(|candidates| {
            candidates
                .iter()
                .find_map(|candidate| gtk::gdk::Cursor::from_name(candidate, None))
        })
    }

    fn resize_cursor_name(handle: ResizeHandle) -> &'static str {
        type RH = ResizeHandle;
        match handle {
            RH::TopLeft | RH::BottomRight => "nwse-resize",
            RH::TopRight | RH::BottomLeft => "nesw-resize",
            RH::TopCenter | RH::BottomCenter => "ns-resize",
            RH::MiddleLeft | RH::MiddleRight => "ew-resize",
        }
    }

    fn set_hover_cursor(&mut self, pos: Vec2D) {
        let Some(widget) = &self.cursor_widget else {
            return;
        };

        let not_renderable = self.preview.as_ref().is_some_and(|p| !p.is_renderable());

        let cursor = if let DragState::Moving { .. } = self.drag_state {
            self.last_drag_state = true;
            if not_renderable {
                self.get_cursor("not-allowed")
            } else {
                self.get_cursor("grabbing")
            }
        } else if let DragState::Resizing { handle, .. } = self.drag_state {
            if not_renderable {
                self.get_cursor("not-allowed")
            } else {
                self.get_cursor(Self::resize_cursor_name(handle))
            }
        } else if matches!(self.drag_state, DragState::None) && self.last_drag_state {
            if let Some(sender) = &self.sender {
                sender.emit(SketchBoardInput::RefreshMouseCursor(pos));
            }
            self.last_drag_state = false;
            None
        } else if let Some(handle) = self.hit_test_handles(pos) {
            self.get_cursor(Self::resize_cursor_name(handle))
        } else {
            None
        };

        if cursor.is_some() {
            widget.set_cursor(cursor.as_ref());
        }
    }

    fn clear_hover_cursor(&self) {
        if let Some(widget) = &self.cursor_widget {
            widget.set_cursor(None);
        }
    }

    pub fn selected_index(&self) -> Option<usize> {
        self.selected_index
    }

    pub fn selected_frame(&self) -> Option<Frame> {
        self.selected_frame
    }

    // Returns the handle under `pos` given the current selection frame.
    pub fn hit_test_handles(&self, pos: Vec2D) -> Option<ResizeHandle> {
        let overlay = self.selection_overlay.as_ref()?;
        let scaled_handle_size = HANDLE_SIZE / overlay.scale.get();
        hit_handle(scaled_handle_size, pos, &overlay.frame)
    }

    // Called by SketchBoard before delivering a BeginDrag event: sets up a move drag.
    pub fn begin_move(&mut self, index: usize, drawable: Box<dyn Drawable>, start_pos: Vec2D) {
        let Some(orig_frame) = Frame::of(drawable.as_ref()) else {
            return;
        };
        self.selected_index = Some(index);
        self.selected_frame = Some(orig_frame);
        self.selection_overlay = None;
        let drawable_clone = drawable.clone_box();
        self.preview = Some(drawable_clone);
        self.last_resize_event = None;
        if let Some(preview) = &mut self.preview {
            preview.set_centered(self.centered);
            preview.set_editing(self.centered);
        }

        self.drag_state = DragState::Moving {
            index,
            original: drawable,
            orig_frame,
        };
        self.drag_start_pos = Some(start_pos);
        self.set_hover_cursor(start_pos);
    }

    // Called by SketchBoard before delivering a BeginDrag event: sets up a resize drag.
    pub fn begin_resize(
        &mut self,
        index: usize,
        drawable: Box<dyn Drawable>,
        handle: ResizeHandle,
        start_pos: Vec2D,
    ) {
        let Some(orig_frame) = Frame::of(drawable.as_ref()) else {
            return;
        };
        self.selected_index = Some(index);
        self.selected_frame = Some(orig_frame);
        self.selection_overlay = None;
        self.preview = Some(drawable.clone_box());
        self.last_resize_event = None;
        self.drag_state = DragState::Resizing {
            index,
            original: drawable,
            handle,
            orig_frame,
        };
        self.drag_start_pos = Some(start_pos);
    }

    fn current_drag_pos(&self, delta: Vec2D) -> Vec2D {
        self.drag_start_pos.map_or(delta, |start| start + delta)
    }

    // The dragged drawable and its frame after the resize drag `event`. A
    // resize happens in the plane of the drawable, so a projected drawable
    // keeps its perspective, and a distortion moves corners on the image.
    fn resized(
        original: &dyn Drawable,
        handle: ResizeHandle,
        orig_frame: &Frame,
        start: Vec2D,
        event: MouseEventMsg,
    ) -> (Box<dyn Drawable>, Frame) {
        if is_distort(event.modifier) {
            let distorted = orig_frame
                .distort(original, handle, event.pos)
                .unwrap_or_else(|| original.clone_box());
            let frame = Frame::of(distorted.as_ref()).unwrap_or(*orig_frame);
            return (distorted, frame);
        }

        let event = MouseEventMsg {
            pos: orig_frame.plane_delta(start, event.pos),
            ..event
        };
        let (new_tl, new_br) = handle.resize(event, orig_frame.tl, orig_frame.br);
        let mut resized = original.clone_box();
        let keep_aspect = event
            .modifier
            .intersects(ModifierType::CONTROL_MASK | ModifierType::SHIFT_MASK);
        resized.resize_bounds_with_segment_target(
            new_tl,
            new_br,
            event.pos,
            keep_aspect,
            handle.segment_target(orig_frame.tl, orig_frame.br),
        );
        (resized, orig_frame.with_bounds(new_tl, new_br))
    }

    fn update_resize_preview(&mut self, event: MouseEventMsg) {
        let DragState::Resizing {
            original,
            handle,
            orig_frame,
            ..
        } = &self.drag_state
        else {
            return;
        };
        let start = self.drag_start_pos.unwrap_or_default();
        let (mut preview, frame) =
            Self::resized(original.as_ref(), *handle, orig_frame, start, event);
        self.centered = event.modifier.intersects(ModifierType::ALT_MASK);
        preview.set_centered(self.centered);
        preview.set_editing(true);
        self.emit_dimensions_update(preview.as_ref());
        self.update_selection_frame(frame);
        self.preview = Some(preview);
    }

    fn emit_dimensions_update(&self, drawable: &dyn Drawable) {
        if let Some(sender) = &self.sender
            && let Some((tl, br)) = drawable.bounds()
        {
            // get it from drawable as it represents the most up-to-date bounds
            sender
                .send(SketchBoardInput::ShapeDimensionsUpdate(br - tl))
                .ok();
        }
    }

    fn update_selection_frame(&mut self, frame: Frame) {
        self.selected_frame = Some(frame);
        self.selection_overlay = Some(SelectionOverlay {
            frame,
            // is updated in draw() to maintain constant on-screen size regardless of zoom level
            scale: Cell::new(1.0),
            centered: self.centered,
        });
    }

    // Select a drawable without starting a drag (e.g. after a commit/replace).
    pub fn set_selection(&mut self, index: usize, frame: Frame) {
        self.selected_index = Some(index);
        self.update_selection_frame(frame);
        self.drag_state = DragState::None;
        self.preview = None;
        self.last_resize_event = None;
    }

    pub fn deselect(&mut self) {
        self.selected_index = None;
        self.selected_frame = None;
        self.selection_overlay = None;
        self.drag_state = DragState::None;
        self.preview = None;
        // Reset cycling state when deselecting
        self.last_click_pos = None;
        self.hit_objects_at_pos.clear();
        self.current_cycle_index = 0;
        self.drag_start_pos = None;
        self.last_resize_event = None;
    }

    // Cycle through overlapping objects at the same position.
    // When Alt+Click is used, this method determines which object to select next.
    // Returns the next object index to cycle through, or None if no objects are at the position.
    pub fn cycle_to_next_object(
        &mut self,
        click_pos: Vec2D,
        hit_indices: Vec<usize>,
    ) -> Option<usize> {
        if hit_indices.is_empty() {
            return None;
        }

        // Check if this is the same position as last click
        if let Some(last_pos) = self.last_click_pos {
            if (last_pos.x - click_pos.x).abs() < 0.1 && (last_pos.y - click_pos.y).abs() < 0.1 {
                // Same position: advance to next object in cycle
                self.current_cycle_index = (self.current_cycle_index + 1) % hit_indices.len();
            } else {
                // Different position: reset cycle
                self.current_cycle_index = 0;
            }
        } else {
            // First time: reset cycle
            self.current_cycle_index = 0;
        }

        // Store position for next cycle check
        self.last_click_pos = Some(click_pos);

        // Return the object at current cycle index
        hit_indices.get(self.current_cycle_index).copied()
    }
}

impl Tool for PointerTool {
    fn get_tool_type(&self) -> Tools {
        Tools::Pointer
    }

    fn get_drawable(&self) -> Option<&dyn Drawable> {
        if let Some(p) = &self.preview {
            Some(p.as_ref())
        } else if let Some(s) = &self.selection_overlay {
            Some(s)
        } else {
            None
        }
    }

    fn input_enabled(&self) -> bool {
        self.input_enabled
    }

    fn set_input_enabled(&mut self, value: bool) {
        self.input_enabled = value;
    }

    fn handle_deactivated(&mut self) -> ToolUpdateResult {
        self.clear_hover_cursor();
        self.deselect();
        ToolUpdateResult::Redraw
    }

    fn handle_key_event(&mut self, event: KeyEventMsg) -> ToolUpdateResult {
        if self.selected_index.is_none() {
            return ToolUpdateResult::Unmodified;
        }

        if matches!(event.key, Key::Alt_L | Key::Alt_R) {
            self.centered = true;
            if let Some(selection_overlay) = &mut self.selection_overlay {
                selection_overlay.centered = true;
            }
            if let Some(mut resize_event) = self.last_resize_event {
                resize_event.modifier = event.modifier | ModifierType::ALT_MASK;
                self.update_resize_preview(resize_event);
            }
            return ToolUpdateResult::Redraw;
        }

        let step = if event.modifier.contains(ModifierType::SHIFT_MASK) {
            APP_CONFIG.read().text_move_length()
        } else {
            1.0
        };

        let delta = match event.key {
            Key::Left => Vec2D::new(-step, 0.0),
            Key::Right => Vec2D::new(step, 0.0),
            Key::Up => Vec2D::new(0.0, -step),
            Key::Down => Vec2D::new(0.0, step),
            _ => return ToolUpdateResult::Unmodified,
        };

        if let Some(sender) = &self.sender {
            sender.emit(SketchBoardInput::NudgeSelection(delta));
            ToolUpdateResult::StopPropagation
        } else {
            ToolUpdateResult::Unmodified
        }
    }

    fn handle_key_release_event(&mut self, event: KeyEventMsg) -> ToolUpdateResult {
        if self.selected_index.is_none() {
            return ToolUpdateResult::Unmodified;
        }

        if matches!(event.key, Key::Alt_L | Key::Alt_R) {
            self.centered = false;
            if let Some(selection_overlay) = &mut self.selection_overlay {
                selection_overlay.centered = false;
            }
            if let Some(mut resize_event) = self.last_resize_event {
                resize_event.modifier = event.modifier & !ModifierType::ALT_MASK;
                self.update_resize_preview(resize_event);
            }
            return ToolUpdateResult::Redraw;
        }

        ToolUpdateResult::Unmodified
    }

    fn handle_mouse_event(&mut self, event: MouseEventMsg) -> ToolUpdateResult {
        if event.button == MouseButton::Middle {
            return ToolUpdateResult::Unmodified;
        }

        // For EndDrag/UpdateDrag, event.pos is the cumulative delta since BeginDrag.
        match event.type_ {
            MouseEventType::BeginDrag => {
                let centered = event.modifier.intersects(ModifierType::ALT_MASK);
                self.centered = centered;
                self.last_resize_event = None;
                if let Some(preview) = &mut self.preview {
                    preview.set_centered(self.centered);
                    preview.set_editing(true);
                }

                ToolUpdateResult::Redraw
            }

            MouseEventType::PointerPos | MouseEventType::Release => {
                self.set_hover_cursor(event.pos);
                ToolUpdateResult::Unmodified
            }

            MouseEventType::UpdateDrag => match &self.drag_state {
                DragState::Moving {
                    original,
                    orig_frame,
                    ..
                } => {
                    let delta = event.pos;
                    let mut preview = original.clone_box();
                    preview.translate(delta);
                    self.emit_dimensions_update(preview.as_ref());
                    let frame = orig_frame.translated(delta);
                    self.update_selection_frame(frame);
                    self.preview = Some(preview);
                    ToolUpdateResult::Redraw
                }
                DragState::Resizing { .. } => {
                    self.last_resize_event = Some(event);
                    self.update_resize_preview(event);
                    ToolUpdateResult::Redraw
                }
                DragState::None => ToolUpdateResult::Unmodified,
            },

            MouseEventType::EndDrag => {
                self.last_resize_event = None;
                let current_pos = self.current_drag_pos(event.pos);
                match std::mem::replace(&mut self.drag_state, DragState::None) {
                    DragState::Moving {
                        index,
                        original,
                        orig_frame,
                    } => {
                        let delta = event.pos;
                        let not_renderable =
                            self.preview.as_ref().is_some_and(|p| !p.is_renderable());
                        let result = if delta.is_zero() || not_renderable {
                            // Click with no movement: just show selection overlay
                            self.update_selection_frame(orig_frame);
                            self.preview = None;
                            ToolUpdateResult::Redraw
                        } else {
                            let mut final_drawable = original;
                            final_drawable.translate(delta);
                            self.update_selection_frame(orig_frame.translated(delta));
                            self.preview = None;
                            ToolUpdateResult::ReplaceDrawable(index, final_drawable)
                        };
                        self.drag_start_pos = None;
                        self.set_hover_cursor(current_pos);
                        result
                    }
                    DragState::Resizing {
                        index,
                        original,
                        handle,
                        orig_frame,
                    } => {
                        let delta = event.pos;
                        let not_renderable =
                            self.preview.as_ref().is_some_and(|p| !p.is_renderable());
                        let resized = if delta.is_zero() || not_renderable {
                            None
                        } else {
                            let start = self.drag_start_pos.unwrap_or_default();
                            Some(Self::resized(
                                original.as_ref(),
                                handle,
                                &orig_frame,
                                start,
                                event,
                            ))
                        };
                        let result = match resized {
                            Some((mut final_drawable, frame)) => {
                                final_drawable.set_centered(false);
                                final_drawable.set_editing(false);
                                self.update_selection_frame(frame);
                                ToolUpdateResult::ReplaceDrawable(index, final_drawable)
                            }
                            None => {
                                self.update_selection_frame(orig_frame);
                                ToolUpdateResult::Redraw
                            }
                        };
                        self.preview = None;
                        self.drag_start_pos = None;
                        self.set_hover_cursor(current_pos);
                        result
                    }
                    DragState::None => {
                        self.drag_start_pos = None;
                        ToolUpdateResult::Unmodified
                    }
                }
            }

            _ => ToolUpdateResult::Unmodified,
        }
    }

    fn set_sender(&mut self, sender: Sender<SketchBoardInput>) {
        self.sender = Some(sender);
    }

    fn set_im_context(&mut self, context: Option<InputContext>) {
        self.cursor_widget = context.map(|ctx| ctx.widget);
        self.clear_hover_cursor();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a drawable that is nothing but its bounds
    #[derive(Clone, Debug)]
    struct Bounds {
        tl: Vec2D,
        br: Vec2D,
    }

    impl Drawable for Bounds {
        fn draw(
            &self,
            _canvas: &mut femtovg::Canvas<femtovg::renderer::OpenGl>,
            _font: FontId,
            _bounds: (Vec2D, Vec2D),
        ) -> Result<()> {
            Ok(())
        }

        fn bounds(&self) -> Option<(Vec2D, Vec2D)> {
            Some((self.tl, self.br))
        }

        fn resize_bounds(&mut self, tl: Vec2D, br: Vec2D, _delta: Vec2D, _keep_aspect: bool) {
            (self.tl, self.br) = ensure_bounding_box(tl, br);
        }
    }

    fn close(a: Vec2D, b: Vec2D) -> bool {
        (a - b).norm() < 1e-2
    }

    fn plain() -> (Box<dyn Drawable>, Frame) {
        let drawable: Box<dyn Drawable> = Box::new(Bounds {
            tl: Vec2D::new(100.0, 100.0),
            br: Vec2D::new(300.0, 200.0),
        });
        let frame = Frame::of(drawable.as_ref()).unwrap();
        (drawable, frame)
    }

    fn drag(delta: Vec2D, modifier: ModifierType) -> MouseEventMsg {
        MouseEventMsg {
            type_: MouseEventType::UpdateDrag,
            button: MouseButton::Primary,
            modifier,
            screen_pos: Vec2D::zero(),
            is_touchpad: false,
            pos: delta,
            n_pressed: 1,
            release: false,
        }
    }

    // the drawable with its top right corner pulled in, so it has a perspective
    fn projected() -> (Box<dyn Drawable>, Frame) {
        let (drawable, frame) = plain();
        let drawable = frame
            .distort(
                drawable.as_ref(),
                ResizeHandle::TopRight,
                Vec2D::new(-20.0, 30.0),
            )
            .unwrap();
        let frame = Frame::of(drawable.as_ref()).unwrap();
        (drawable, frame)
    }

    #[test]
    fn distorting_a_corner_moves_only_that_corner() {
        let (drawable, frame) = plain();
        let delta = Vec2D::new(-20.0, 30.0);
        let distorted = frame
            .distort(drawable.as_ref(), ResizeHandle::TopRight, delta)
            .unwrap();
        let (before, after) = (frame.quad(), Frame::of(distorted.as_ref()).unwrap().quad());
        assert!(close(after[1], before[1] + delta));
        for i in [0, 2, 3] {
            assert!(close(after[i], before[i]), "corner {i}");
        }
    }

    #[test]
    fn distorting_an_edge_moves_both_of_its_corners() {
        let (drawable, frame) = plain();
        let delta = Vec2D::new(15.0, -10.0);
        let distorted = frame
            .distort(drawable.as_ref(), ResizeHandle::MiddleLeft, delta)
            .unwrap();
        let (before, after) = (frame.quad(), Frame::of(distorted.as_ref()).unwrap().quad());
        for i in [3, 0] {
            assert!(close(after[i], before[i] + delta), "corner {i}");
        }
        for i in [1, 2] {
            assert!(close(after[i], before[i]), "corner {i}");
        }
    }

    #[test]
    fn a_distortion_stops_where_the_projection_is_rejected() {
        let (drawable, frame) = plain();
        // past the opposite corner, which would fold the quad
        let delta = Vec2D::new(-400.0, 300.0);
        let distorted = frame
            .distort(drawable.as_ref(), ResizeHandle::TopRight, delta)
            .unwrap();
        let after = Frame::of(distorted.as_ref()).unwrap();
        assert!(Projected::projection(after.tl, after.br, &after.quad()).is_some());
        let moved = after.quad()[1] - frame.quad()[1];
        assert!(moved.norm() > 1.0 && moved.norm() < delta.norm());
    }

    #[test]
    fn a_resize_follows_the_pointer_and_keeps_the_projection() {
        let (drawable, frame) = projected();
        let handle = ResizeHandle::MiddleRight;
        let start = frame.handle_pos(handle);
        // along the projected right edge's normal is not a plane direction, so
        // only check where the handle lands along the plane's x axis
        let delta = Vec2D::new(40.0, 0.0);
        let (resized, after) = PointerTool::resized(
            drawable.as_ref(),
            handle,
            &frame,
            start,
            drag(delta, ModifierType::empty()),
        );

        assert_eq!(after.h, frame.h);
        assert_eq!(Frame::of(resized.as_ref()).unwrap(), after);
        // the dragged edge moved in the plane only, and the pointer is on its
        // line through the handle
        assert_eq!(after.tl, frame.tl);
        assert!(after.br.x > frame.br.x);
        let h = frame.h.unwrap();
        let inverse = h.inverse().unwrap();
        assert!((inverse.apply(start + delta).x - after.br.x).abs() < 1e-2);
    }

    #[test]
    fn a_resize_past_the_horizon_of_the_plane_is_ignored() {
        let (_, frame) = projected();
        let start = frame.handle_pos(ResizeHandle::MiddleRight);
        let far = Vec2D::new(1e6, 0.0);
        assert_eq!(frame.plane_delta(start, far), Vec2D::zero());
    }

    #[test]
    fn only_ctrl_and_shift_together_distort() {
        assert!(is_distort(
            ModifierType::CONTROL_MASK | ModifierType::SHIFT_MASK
        ));
        assert!(!is_distort(ModifierType::CONTROL_MASK));
        assert!(!is_distort(ModifierType::SHIFT_MASK));
    }
}
