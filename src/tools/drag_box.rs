use femtovg::{Color, Paint, Path};
use relm4::Sender;
use relm4::gtk::gdk::ModifierType;

use crate::{
    configuration::APP_CONFIG,
    math::{
        Vec2D, get_closest_aspect_ratio,
        homography::{Quad, outset_quad},
    },
    sketch_board::{MouseEventMsg, SketchBoardInput},
    tools::projected::quad_path,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DragBox {
    pub top_left: Vec2D,
    pub size: Vec2D,
    pub centered: bool,
    pub keep_aspect: bool,
    pub delta: Vec2D,
}

impl DragBox {
    pub fn from_origin_delta(
        origin: Vec2D,
        orig_size: Vec2D,
        event: &MouseEventMsg,
        sender: &Sender<SketchBoardInput>,
    ) -> Self {
        let drag_box = Self::from_delta(origin, orig_size, event.pos, event.modifier);
        sender
            .send(SketchBoardInput::ShapeDimensionsUpdate(drag_box.size))
            .ok();
        drag_box
    }

    // the box itself, without reporting its dimensions
    pub fn from_delta(
        origin: Vec2D,
        orig_size: Vec2D,
        delta: Vec2D,
        modifier: ModifierType,
    ) -> Self {
        let aspect = modifier.intersects(ModifierType::SHIFT_MASK);
        let keep_aspect = modifier.intersects(ModifierType::CONTROL_MASK);
        let centered = modifier.intersects(ModifierType::ALT_MASK);

        let mut new_size = delta;
        let sign_x = new_size.x.signum();
        let sign_y = new_size.y.signum();
        let width = new_size.x.abs();
        let height = new_size.y.abs();

        if keep_aspect {
            let aspect_ratio =
                if orig_size.x.abs() <= f32::EPSILON || orig_size.y.abs() <= f32::EPSILON {
                    // start with square
                    1.0
                } else {
                    orig_size.x.abs() / orig_size.y.abs()
                };
            if width >= height * aspect_ratio {
                new_size.y = width / aspect_ratio * sign_y;
            } else {
                new_size.x = height * aspect_ratio * sign_x;
            }
        } else if aspect && new_size.y.abs() > f32::EPSILON {
            let aspect_ratio = width / height;
            let config = APP_CONFIG.read();
            let closest_aspect_ratio =
                get_closest_aspect_ratio(aspect_ratio, config.aspect_ratios());
            let aspect_ratio = closest_aspect_ratio.0 / closest_aspect_ratio.1;
            let (width, height) = if height > width / aspect_ratio {
                (height * aspect_ratio, height)
            } else {
                (width, width / aspect_ratio)
            };
            new_size.x = width * sign_x;
            new_size.y = height * sign_y;
        }

        let size_factor = if centered { 2.0 } else { 1.0 };
        new_size *= size_factor;

        let top_left = if centered {
            origin.min(origin - new_size.abs() / size_factor)
        } else {
            origin.min(origin + new_size)
        };

        Self {
            top_left,
            size: new_size.abs(),
            centered,
            keep_aspect: aspect || keep_aspect,
            delta,
        }
    }

    pub fn middle(&self) -> Vec2D {
        self.top_left + self.size * 0.5
    }
}

fn inner_color() -> Paint {
    Paint::color(Color::white())
}

fn outer_color() -> Paint {
    Paint::color(Color::rgb(70u8, 130u8, 180u8))
}

pub fn draw_center_marker(canvas: &mut femtovg::Canvas<femtovg::renderer::OpenGl>, center: Vec2D) {
    // in inverse zoom scale so the visual size stays constant on screen
    let scale = canvas.transform().average_scale().max(f32::EPSILON);

    let radius = 4.0 / scale;
    let mut path = Path::new();
    path.circle(center.x, center.y, radius);
    canvas.fill_path(&path, &inner_color());
    canvas.stroke_path(&path, &outer_color().with_line_width(1.5 / scale));
}

// in inverse zoom scale so the visual size stays constant on screen
fn marker_line_width(canvas: &femtovg::Canvas<femtovg::renderer::OpenGl>) -> f32 {
    1.5 / canvas.transform().average_scale().max(f32::EPSILON)
}

// a marker is `inner` in the inner color, filled or stroked, and a stroke
// along `outer` in the outer color
fn draw_marker(
    canvas: &mut femtovg::Canvas<femtovg::renderer::OpenGl>,
    inner: &Path,
    outer: &Path,
    filled: bool,
) {
    let line_width = marker_line_width(canvas);
    let border_paint = inner_color().with_line_width(line_width);
    if filled {
        canvas.fill_path(inner, &border_paint);
    } else {
        canvas.stroke_path(inner, &border_paint);
    }
    canvas.stroke_path(outer, &outer_color().with_line_width(line_width));
}

pub fn draw_rect_marker(
    canvas: &mut femtovg::Canvas<femtovg::renderer::OpenGl>,
    top_left: Vec2D,
    size: Vec2D,
    filled: bool,
) {
    let line_width = marker_line_width(canvas);
    let mut inner = Path::new();
    inner.rect(top_left.x, top_left.y, size.x, size.y);

    let mut tl = top_left;
    let mut size = size;

    // set-out to draw the border outside the original rectangle, e.g. for crop
    if !filled {
        tl = tl - line_width;
        size = size + 2.0 * line_width;
    }

    let mut outer = Path::new();
    outer.rect(tl.x, tl.y, size.x, size.y);
    draw_marker(canvas, &inner, &outer, filled);
}

// like draw_rect_marker, for a strictly convex quad
pub fn draw_quad_marker(
    canvas: &mut femtovg::Canvas<femtovg::renderer::OpenGl>,
    quad: &Quad,
    filled: bool,
) {
    let outer = if filled {
        *quad
    } else {
        outset_quad(quad, marker_line_width(canvas))
    };
    draw_marker(canvas, &quad_path(quad), &quad_path(&outer), filled);
}
