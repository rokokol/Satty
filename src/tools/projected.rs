use std::cell::{Cell, RefCell};

use anyhow::Result;
use femtovg::{
    Canvas, Color, FontId, ImageFlags, ImageId, Paint, Path, PixelFormat, RenderTarget,
    Transform2D, imgref::ImgVec, renderer::OpenGl, rgb::RGBA8,
};

use crate::{
    math::{
        Vec2D, ensure_bounding_box,
        homography::{Homography, Quad, affine_from_triangles},
    },
    style::Style,
};

use super::{Drawable, PrepareContext, RenderingMode};

/// A drawable shown through a projective transformation.
///
/// The inner drawable keeps its own geometry in a plane of its own, and moves
/// and resizes in that plane. femtovg only draws affine transformations, so
/// the inner drawable is drawn into an offscreen image first, and the image is
/// then drawn as a grid of triangles, each with the affine transformation
/// that matches the projection at its corners.
#[derive(Debug)]
pub struct Projected {
    inner: Box<dyn Drawable>,
    // from the plane of `inner` onto the image
    h: Homography,
    // the offscreen image of `inner` for the current frame, and the plane
    // rectangle it shows
    texture: Cell<Option<(ImageId, Vec2D, Vec2D)>>,
    effect: RefCell<Option<EffectCover>>,
}

// for an effect on the screenshot, a copy of the inner drawable that covers
// the bounds of the quad on the image, with those bounds; the effect itself is
// not projected, only the area it shows in
type EffectCover = ((Vec2D, Vec2D), Box<dyn Drawable>);

impl Clone for Projected {
    fn clone(&self) -> Self {
        Self::from_parts(self.inner.clone_box(), self.h)
    }
}

// femtovg refuses offscreen images larger than this on either side
const MAX_TEXTURE_SIZE: f32 = 8192.0;
// the most a triangle may differ from the true projection, in screen pixels
const MAX_MESH_ERROR: f32 = 0.5;
const MAX_MESH_CELLS: usize = 32;
// the denominator of the projection at a corner of the offscreen image is at
// least this fraction of its value at the plane bounds, so the image stays well
// in front of the horizon
const MIN_DENOMINATOR_RATIO: f64 = 0.2;
// a nearly flat quad puts one corner almost on the horizon, where a single
// texel covers a large part of the screen; this keeps the nearest corner at
// most this many times nearer than the farthest one
const MAX_DEPTH_RATIO: f64 = 10.0;

impl Projected {
    /// `drawable` shown so that its bounds fill `quad`. A drawable that is
    /// already projected keeps its plane and only changes the projection.
    /// `None` when `quad` is not strictly convex, or the drawable has no area
    /// or is not drawn in the stack of annotations.
    pub fn project(drawable: &dyn Drawable, quad: &Quad) -> Option<Box<dyn Drawable>> {
        let inner = match drawable.as_any().downcast_ref::<Projected>() {
            Some(projected) => projected.inner.clone_box(),
            None => drawable.clone_box(),
        };
        if !matches!(
            inner.get_rendering_mode(),
            RenderingMode::Default | RenderingMode::Blur
        ) {
            return None;
        }
        let (tl, br) = inner.bounds()?;
        let h = Self::projection(tl, br, quad)?;
        Some(Box::new(Self::from_parts(inner, h)))
    }

    /// The projection of the plane rectangle from `tl` to `br` onto `quad`, if
    /// a drawable may be shown through it.
    pub fn projection(tl: Vec2D, br: Vec2D, quad: &Quad) -> Option<Homography> {
        Homography::from_rect_to_quad(tl, br, quad)
            .filter(|h| h.depth_ratio(tl, br) <= MAX_DEPTH_RATIO)
    }

    /// The drawable without its projection, and the projection.
    pub fn into_parts(self) -> (Box<dyn Drawable>, Homography) {
        (self.inner, self.h)
    }

    /// `inner` shown through `h`, as it was before `into_parts`.
    pub fn from_parts(inner: Box<dyn Drawable>, h: Homography) -> Self {
        Self {
            inner,
            h,
            // the image is deleted after the frame it was made for
            texture: Cell::new(None),
            effect: RefCell::new(None),
        }
    }

    fn quad(&self) -> Option<Quad> {
        let (tl, br) = self.inner.bounds()?;
        Some(self.h.map_rect(tl, br))
    }

    // the plane rectangle the offscreen image covers: the bounds of `inner`
    // with room for strokes and text that reach past them
    fn texture_rect(&self) -> Option<(Vec2D, Vec2D)> {
        let (tl, br) = self.inner.bounds()?;
        let size = br - tl;
        let reference = self.h.denominator((tl + br) * 0.5);
        let mut padding = size.x.max(size.y) * 0.15 + 16.0;
        // a strong perspective puts the horizon close to the bounds, and the
        // padding must not reach it
        for _ in 0..8 {
            let (ptl, pbr) = (tl - padding, br + padding);
            let corners = [ptl, Vec2D::new(pbr.x, ptl.y), pbr, Vec2D::new(ptl.x, pbr.y)];
            if corners
                .iter()
                .all(|c| self.h.denominator(*c) > reference * MIN_DENOMINATOR_RATIO)
            {
                return Some((ptl, pbr - ptl));
            }
            padding *= 0.5;
        }
        Some((tl, size))
    }
}

/// The number of grid cells along each side that keeps the triangles of the
/// plane rectangle at `tl` of `size` within `max_error` of `h`, measured after
/// a further scale by `scale`.
fn mesh_cells(h: &Homography, tl: Vec2D, size: Vec2D, scale: f32, max_error: f32) -> usize {
    // the two triangles of a cell share its diagonal, and they are furthest
    // from the projection at its middle; the denominator is linear, so the
    // most distorted cell is one at a corner
    let error = |nx: usize, ny: usize| {
        let cell = Vec2D::new(size.x / nx as f32, size.y / ny as f32);
        [(0, 0), (nx - 1, 0), (nx - 1, ny - 1), (0, ny - 1)]
            .iter()
            .map(|&(i, j)| {
                let p00 = tl + Vec2D::new(cell.x * i as f32, cell.y * j as f32);
                let p11 = p00 + cell;
                let middle = (h.apply(p00) + h.apply(p11)) * 0.5;
                (h.apply(p00 + cell * 0.5) - middle).norm()
            })
            .fold(0.0, f32::max)
            * scale
    };

    // the error has a term in the product of the cell's width and height, so
    // a split along one axis alone does not get far
    (1..MAX_MESH_CELLS)
        .find(|&n| error(n, n) <= max_error)
        .unwrap_or(MAX_MESH_CELLS)
}

/// The closed outline of `quad`.
pub fn quad_path(quad: &Quad) -> Path {
    let mut path = Path::new();
    path.move_to(quad[0].x, quad[0].y);
    for p in &quad[1..] {
        path.line_to(p.x, p.y);
    }
    path.close();
    path
}

impl Drawable for Projected {
    fn prepare(&self, canvas: &mut Canvas<OpenGl>, context: &mut PrepareContext) -> Result<()> {
        self.texture.set(None);
        self.inner.prepare(canvas, context)?;
        // an effect reads the screen it is drawn on, see draw_baselayer
        if self.inner.get_rendering_mode() == RenderingMode::Blur {
            return Ok(());
        }
        let Some((tl, size)) = self.texture_rect() else {
            return Ok(());
        };
        if size.x <= 0.0 || size.y <= 0.0 {
            return Ok(());
        }

        // as many texels as the most stretched part of the projection shows
        // screen pixels
        let canvas_scale = canvas.transform().average_scale().max(f32::EPSILON);
        let br = tl + size;
        let stretch = [tl, Vec2D::new(br.x, tl.y), br, Vec2D::new(tl.x, br.y)]
            .iter()
            .map(|c| self.h.local_scale(*c))
            .fold(f32::EPSILON, f32::max);
        let mut scale = canvas_scale * stretch;
        scale = scale.min(MAX_TEXTURE_SIZE / size.x.max(size.y));
        let width = (size.x * scale).ceil().max(1.0);
        let height = (size.y * scale).ceil().max(1.0);

        // offscreen images hold premultiplied colors upside down, see
        // femtovg's Canvas::render_shadow
        let image = canvas.create_image_empty(
            width as usize,
            height as usize,
            PixelFormat::Rgba8,
            ImageFlags::PREMULTIPLIED | ImageFlags::FLIP_Y,
        )?;
        context.images.push(image);

        canvas.save();
        canvas.set_render_target(RenderTarget::Image(image));
        canvas.reset_transform();
        canvas.reset_scissor();
        canvas.clear_rect(
            0,
            0,
            width as u32,
            height as u32,
            Color::rgbaf(0.0, 0.0, 0.0, 0.0),
        );
        canvas.set_transform(&Transform2D([
            scale,
            0.0,
            0.0,
            scale,
            -tl.x * scale,
            -tl.y * scale,
        ]));
        let result = self.inner.draw(canvas, context.font, context.bounds);
        canvas.restore();
        canvas.set_render_target(context.target);
        result?;

        self.texture.set(Some((image, tl, size)));
        Ok(())
    }

    fn draw(
        &self,
        canvas: &mut Canvas<OpenGl>,
        _font: FontId,
        _bounds: (Vec2D, Vec2D),
    ) -> Result<()> {
        // without a prepared image there is nothing to project
        let Some((image, tl, size)) = self.texture.get() else {
            return Ok(());
        };

        let to_screen = canvas.transform();
        let canvas_scale = to_screen.average_scale().max(f32::EPSILON);
        let cells = mesh_cells(&self.h, tl, size, canvas_scale, MAX_MESH_ERROR);
        let plane: Vec<Vec2D> = (0..=cells)
            .flat_map(|j| {
                (0..=cells).map(move |i| {
                    tl + Vec2D::new(
                        size.x * i as f32 / cells as f32,
                        size.y * j as f32 / cells as f32,
                    )
                })
            })
            .collect();
        let image_points: Vec<Vec2D> = plane.iter().map(|p| self.h.apply(*p)).collect();
        let screen: Vec<Vec2D> = image_points
            .iter()
            .map(|p| {
                let (x, y) = to_screen.transform_point(p.x, p.y);
                Vec2D::new(x, y)
            })
            .collect();
        let (width, height) = (canvas.width() as f32, canvas.height() as f32);
        let paint =
            Paint::image(image, tl.x, tl.y, size.x, size.y, 0.0, 1.0).with_anti_alias(false);

        for j in 0..cells {
            for i in 0..cells {
                let index = |i: usize, j: usize| j * (cells + 1) + i;
                let (v00, v10) = (index(i, j), index(i + 1, j));
                let (v01, v11) = (index(i, j + 1), index(i + 1, j + 1));
                for corners in [[v00, v10, v11], [v00, v11, v01]] {
                    // femtovg skips a path off the canvas only after it has
                    // flattened it, which is most of the cost when zoomed in
                    let on_screen = corners.map(|v| screen[v]);
                    let min = on_screen.iter().fold(on_screen[0], |a, b| a.min(*b));
                    let max = on_screen.iter().fold(on_screen[0], |a, b| a.max(*b));
                    if max.x < 0.0 || max.y < 0.0 || min.x > width || min.y > height {
                        continue;
                    }

                    let triangle = corners.map(|v| plane[v]);
                    let Some(affine) =
                        affine_from_triangles(&triangle, &corners.map(|v| image_points[v]))
                    else {
                        continue;
                    };
                    let mut path = Path::new();
                    path.move_to(triangle[0].x, triangle[0].y);
                    path.line_to(triangle[1].x, triangle[1].y);
                    path.line_to(triangle[2].x, triangle[2].y);
                    path.close();

                    canvas.save();
                    canvas.set_transform(&Transform2D(affine));
                    canvas.fill_path(&path, &paint);
                    canvas.restore();
                }
            }
        }
        Ok(())
    }

    fn draw_baselayer(
        &self,
        canvas: &mut Canvas<OpenGl>,
        image: &ImgVec<RGBA8>,
        font: FontId,
        bounds: (Vec2D, Vec2D),
    ) -> Result<()> {
        if self.inner.get_rendering_mode() != RenderingMode::Blur {
            return self.draw(canvas, font, bounds);
        }
        let (Some(quad), Some(quad_bounds)) = (self.quad(), self.bounds()) else {
            return Ok(());
        };

        let mut effect = self.effect.borrow_mut();
        if effect.as_ref().map(|(b, _)| *b) != Some(quad_bounds) {
            let mut cover = self.inner.clone_box();
            cover.resize_bounds(quad_bounds.0, quad_bounds.1, Vec2D::zero(), false);
            *effect = Some((quad_bounds, cover));
        }
        let Some((_, cover)) = effect.as_ref() else {
            return Ok(());
        };
        let Some(paint) = cover.effect_paint(canvas, image, bounds)? else {
            return Ok(());
        };

        // the effect only exists on the screenshot, like the unprojected one
        let (tl, br) = bounds;
        canvas.save();
        canvas.intersect_scissor(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
        canvas.fill_path(&quad_path(&quad), &paint);
        canvas.restore();
        Ok(())
    }

    fn perspective(&self) -> Option<(Homography, (Vec2D, Vec2D))> {
        Some((self.h, self.inner.bounds()?))
    }

    fn handle_undo(&mut self) {
        self.inner.handle_undo();
    }

    fn handle_redo(&mut self) {
        self.inner.handle_redo();
    }

    fn get_rendering_mode(&self) -> RenderingMode {
        self.inner.get_rendering_mode()
    }

    fn is_renderable(&self) -> bool {
        self.inner.is_renderable()
    }

    fn bounds_only_valid_after_redraw(&self) -> bool {
        self.inner.bounds_only_valid_after_redraw()
    }

    fn bounds(&self) -> Option<(Vec2D, Vec2D)> {
        let (tl, br) = self.inner.bounds()?;
        let quad = self.h.map_rect(tl, br);
        let min = quad.iter().fold(quad[0], |a, b| a.min(*b));
        let max = quad.iter().fold(quad[0], |a, b| a.max(*b));
        Some(ensure_bounding_box(min, max))
    }

    fn hit_test(&self, pos: Vec2D, tolerance: f32) -> bool {
        let Some(inverse) = self.h.inverse() else {
            return false;
        };
        if inverse.denominator(pos) <= 0.0 {
            return false;
        }
        let plane_pos = inverse.apply(pos);
        // the tolerance is meant on the image, and the plane is stretched
        let tolerance = tolerance / self.h.local_scale(plane_pos).max(f32::EPSILON);
        self.inner.hit_test(plane_pos, tolerance)
    }

    fn translate(&mut self, delta: Vec2D) {
        self.h = self.h.then_translate(delta);
    }

    // the corners are in the plane of the inner drawable, see `perspective`
    fn resize_bounds(&mut self, tl: Vec2D, br: Vec2D, delta: Vec2D, keep_aspect: bool) {
        self.inner.resize_bounds(tl, br, delta, keep_aspect);
    }

    fn resize_bounds_with_segment_target(
        &mut self,
        tl: Vec2D,
        br: Vec2D,
        delta: Vec2D,
        keep_aspect: bool,
        target: Option<crate::math::SegmentResizeTarget>,
    ) {
        self.inner
            .resize_bounds_with_segment_target(tl, br, delta, keep_aspect, target);
    }

    fn get_style(&self) -> Option<&Style> {
        self.inner.get_style()
    }

    fn get_style_mut(&mut self) -> Option<&mut Style> {
        *self.effect.get_mut() = None;
        self.inner.get_style_mut()
    }

    fn set_centered(&mut self, centered: bool) {
        self.inner.set_centered(centered);
    }

    fn set_editing(&mut self, editing: bool) {
        *self.effect.get_mut() = None;
        self.inner.set_editing(editing);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a trapezoid whose right side is `ratio` times shorter than its left
    // side, so the right side is that many times farther away
    fn trapezoid(ratio: f32) -> Quad {
        let half = 50.0 / ratio;
        [
            Vec2D::new(0.0, 0.0),
            Vec2D::new(100.0, 50.0 - half),
            Vec2D::new(100.0, 50.0 + half),
            Vec2D::new(0.0, 100.0),
        ]
    }

    // the largest distance between the triangles of a grid and the true
    // projection, over points spread through every cell
    fn mesh_error(h: &Homography, tl: Vec2D, size: Vec2D, cells: usize) -> f32 {
        let vertex = |i: usize, j: usize| {
            tl + Vec2D::new(
                size.x * i as f32 / cells as f32,
                size.y * j as f32 / cells as f32,
            )
        };
        let mut error = 0.0f32;
        for j in 0..cells {
            for i in 0..cells {
                let (p00, p10) = (vertex(i, j), vertex(i + 1, j));
                let (p01, p11) = (vertex(i, j + 1), vertex(i + 1, j + 1));
                for triangle in [[p00, p10, p11], [p00, p11, p01]] {
                    let [a, b, c, d, e, f] =
                        affine_from_triangles(&triangle, &triangle.map(|p| h.apply(p))).unwrap();
                    // barycentric samples inside the triangle
                    for (u, v) in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (4.0, 1.0), (1.0, 4.0)] {
                        let w = 6.0 - u - v;
                        let p = (triangle[0] * u + triangle[1] * v + triangle[2] * w) / 6.0;
                        let mesh = Vec2D::new(a * p.x + c * p.y + e, b * p.x + d * p.y + f);
                        error = error.max((mesh - h.apply(p)).norm());
                    }
                }
            }
        }
        error
    }

    #[test]
    fn an_affine_projection_needs_no_grid() {
        let (tl, br) = (Vec2D::zero(), Vec2D::new(100.0, 100.0));
        let parallelogram = [
            Vec2D::new(0.0, 0.0),
            Vec2D::new(100.0, 20.0),
            Vec2D::new(130.0, 120.0),
            Vec2D::new(30.0, 100.0),
        ];
        let h = Homography::from_rect_to_quad(tl, br, &parallelogram).unwrap();
        assert_eq!(mesh_cells(&h, tl, br - tl, 4.0, 0.25), 1);
    }

    #[test]
    fn the_grid_stays_within_the_error() {
        let (tl, size) = (Vec2D::new(-30.0, -20.0), Vec2D::new(400.0, 250.0));
        let quad = [
            Vec2D::new(-10.0, 0.0),
            Vec2D::new(380.0, 60.0),
            Vec2D::new(340.0, 210.0),
            Vec2D::new(20.0, 260.0),
        ];
        let h = Homography::from_rect_to_quad(tl, tl + size, &quad).unwrap();
        // a larger scale needs more cells than the grid is allowed to have
        for scale in [0.5, 1.0] {
            let cells = mesh_cells(&h, tl, size, scale, 0.25);
            let error = mesh_error(&h, tl, size, cells) * scale;
            assert!(error <= 0.25, "{cells} at {scale}: {error}");
            // and it is not finer than it has to be
            if cells > 1 {
                assert!(mesh_error(&h, tl, size, cells - 1) * scale > 0.25 * 0.5);
            }
        }
    }

    #[test]
    fn projection_rejects_a_quad_too_deep_to_draw() {
        let (tl, br) = (Vec2D::zero(), Vec2D::new(100.0, 100.0));
        assert!(Projected::projection(tl, br, &trapezoid(4.0)).is_some());
        assert!(Projected::projection(tl, br, &trapezoid(MAX_DEPTH_RATIO as f32 * 2.0)).is_none());
    }
}
