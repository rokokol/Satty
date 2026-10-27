use super::Vec2D;

/// A projective transformation of the plane, as a 3x3 matrix in row-major order.
///
/// Points are mapped as `(x, y, 1)` column vectors, followed by a division by
/// the third coordinate. The matrix is kept in `f64`, because the division
/// amplifies rounding errors of nearly degenerate quadrilaterals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography([f64; 9]);

/// The corners of a quadrilateral in the order top left, top right, bottom
/// right, bottom left of the rectangle it is the image of.
pub type Quad = [Vec2D; 4];

// below this fraction of the product of the lengths, two vectors count as
// parallel, so that a nearly flat corner does not produce a huge matrix
const PARALLEL_TOLERANCE: f64 = 1e-6;

fn cross(a: (f64, f64), b: (f64, f64)) -> f64 {
    a.0 * b.1 - a.1 * b.0
}

fn sub(a: Vec2D, b: Vec2D) -> (f64, f64) {
    (a.x as f64 - b.x as f64, a.y as f64 - b.y as f64)
}

fn length(a: (f64, f64)) -> f64 {
    a.0.hypot(a.1)
}

impl Homography {
    pub fn identity() -> Self {
        Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    /// The transformation that maps the corners of the rectangle from `tl` to
    /// `br` onto the corners of `quad`, or `None` if `quad` is not a strictly
    /// convex quadrilateral with the same orientation as the rectangle.
    pub fn from_rect_to_quad(tl: Vec2D, br: Vec2D, quad: &Quad) -> Option<Self> {
        let (w, h) = (br.x as f64 - tl.x as f64, br.y as f64 - tl.y as f64);
        if w <= 0.0 || h <= 0.0 || !is_convex_quad(quad) {
            return None;
        }
        let to_unit_square = Self([
            1.0 / w,
            0.0,
            -tl.x as f64 / w,
            0.0,
            1.0 / h,
            -tl.y as f64 / h,
            0.0,
            0.0,
            1.0,
        ]);
        Some(Self::from_unit_square_to_quad(quad).compose(&to_unit_square))
    }

    // Heckbert, "Fundamentals of Texture Mapping and Image Warping", 1989,
    // section 2.2.3; the quad is convex, so the denominator is not zero
    fn from_unit_square_to_quad(quad: &Quad) -> Self {
        let [x0, x1, x2, x3] = quad.map(|p| p.x as f64);
        let [y0, y1, y2, y3] = quad.map(|p| p.y as f64);
        let (dx1, dx2, dx3) = (x1 - x2, x3 - x2, x0 - x1 + x2 - x3);
        let (dy1, dy2, dy3) = (y1 - y2, y3 - y2, y0 - y1 + y2 - y3);
        let den = dx1 * dy2 - dx2 * dy1;
        let g = (dx3 * dy2 - dx2 * dy3) / den;
        let h = (dx1 * dy3 - dx3 * dy1) / den;
        Self([
            x1 - x0 + g * x1,
            x3 - x0 + h * x3,
            x0,
            y1 - y0 + g * y1,
            y3 - y0 + h * y3,
            y0,
            g,
            h,
            1.0,
        ])
    }

    // the transformation that applies `other` first and `self` second
    fn compose(&self, other: &Self) -> Self {
        let (a, b) = (&self.0, &other.0);
        let mut m = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                m[row * 3 + col] = (0..3).map(|k| a[row * 3 + k] * b[k * 3 + col]).sum();
            }
        }
        Self(m)
    }

    pub fn apply(&self, p: Vec2D) -> Vec2D {
        let m = &self.0;
        let (x, y) = (p.x as f64, p.y as f64);
        let w = m[6] * x + m[7] * y + m[8];
        Vec2D::new(
            ((m[0] * x + m[1] * y + m[2]) / w) as f32,
            ((m[3] * x + m[4] * y + m[5]) / w) as f32,
        )
    }

    pub fn inverse(&self) -> Option<Self> {
        let [a, b, c, d, e, f, g, h, i] = self.0;
        let adjugate = [
            e * i - f * h,
            c * h - b * i,
            b * f - c * e,
            f * g - d * i,
            a * i - c * g,
            c * d - a * f,
            d * h - e * g,
            b * g - a * h,
            a * e - b * d,
        ];
        let det = a * adjugate[0] + b * adjugate[3] + c * adjugate[6];
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        Some(Self(adjugate.map(|v| v / det)))
    }

    /// The image of the rectangle from `tl` to `br`.
    pub fn map_rect(&self, tl: Vec2D, br: Vec2D) -> Quad {
        [tl, Vec2D::new(br.x, tl.y), br, Vec2D::new(tl.x, br.y)].map(|p| self.apply(p))
    }

    /// This transformation followed by a translation by `delta`.
    pub fn then_translate(&self, delta: Vec2D) -> Self {
        let (dx, dy) = (delta.x as f64, delta.y as f64);
        Self([1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0]).compose(self)
    }

    /// The homogeneous coordinate of the image of `p` before the division. It
    /// is positive on the side of the horizon line where the rectangle that
    /// made this transformation lies, and points on the other side have no
    /// meaningful image.
    pub fn denominator(&self, p: Vec2D) -> f64 {
        let m = &self.0;
        m[6] * p.x as f64 + m[7] * p.y as f64 + m[8]
    }

    /// How many times nearer to the viewer the nearest corner of the rectangle
    /// from `tl` to `br` is than the farthest one, as the ratio of the largest
    /// to the smallest denominator. It is 1 for an affine transformation.
    pub fn depth_ratio(&self, tl: Vec2D, br: Vec2D) -> f64 {
        let w =
            [tl, Vec2D::new(br.x, tl.y), br, Vec2D::new(tl.x, br.y)].map(|p| self.denominator(p));
        let max = w.iter().copied().fold(f64::MIN, f64::max);
        let min = w.iter().copied().fold(f64::MAX, f64::min);
        if min <= 0.0 { f64::INFINITY } else { max / min }
    }

    /// How much a short segment at `p` is stretched at most, in any direction.
    pub fn local_scale(&self, p: Vec2D) -> f32 {
        let m = &self.0;
        let w = self.denominator(p);
        let image = self.apply(p);
        let (x, y) = (image.x as f64, image.y as f64);
        // the Jacobian of the division by w
        let j = [
            (m[0] - x * m[6]) / w,
            (m[1] - x * m[7]) / w,
            (m[3] - y * m[6]) / w,
            (m[4] - y * m[7]) / w,
        ];
        // the largest singular value, from the eigenvalues of J^T J
        let trace = j.iter().map(|v| v * v).sum::<f64>();
        let det = j[0] * j[3] - j[1] * j[2];
        let discriminant = (trace * trace - 4.0 * det * det).max(0.0);
        ((trace + discriminant.sqrt()) / 2.0).sqrt() as f32
    }
}

/// `quad` with each edge moved outwards by `distance`.
pub fn outset_quad(quad: &Quad, distance: f32) -> Quad {
    // the moved edge i as a point on it and its direction
    let edges: [(Vec2D, Vec2D); 4] = std::array::from_fn(|i| {
        let (a, b) = (quad[i], quad[(i + 1) % 4]);
        let direction = b - a;
        // with y pointing down, the left normal of a clockwise edge points outwards
        let normal = Vec2D::new(direction.y, -direction.x) / direction.norm();
        (a + normal * distance, direction)
    });
    std::array::from_fn(|i| {
        let (p, r) = edges[(i + 3) % 4];
        let (q, s) = edges[i];
        // the edges of a convex quad are never parallel at a shared corner
        p + r * ((q - p).cross(s) / r.cross(s))
    })
}

/// Whether `quad` is strictly convex and keeps the orientation of the
/// rectangle it was made from, so that a transformation onto it neither
/// mirrors nor folds the content.
pub fn is_convex_quad(quad: &Quad) -> bool {
    (0..4).all(|i| {
        let e1 = sub(quad[(i + 1) % 4], quad[i]);
        let e2 = sub(quad[(i + 2) % 4], quad[(i + 1) % 4]);
        // with y pointing down, the corners of a rectangle turn clockwise on
        // screen, which is a positive cross product
        cross(e1, e2) > PARALLEL_TOLERANCE * length(e1) * length(e2)
    })
}

/// The affine transformation `[a, b, c, d, e, f]` that maps the triangle `src`
/// onto `dst`, in the layout of `femtovg::Transform2D`: a point goes to
/// `(a * x + c * y + e, b * x + d * y + f)`. `None` for a degenerate `src`.
pub fn affine_from_triangles(src: &[Vec2D; 3], dst: &[Vec2D; 3]) -> Option<[f32; 6]> {
    let (u1, u2) = (sub(src[1], src[0]), sub(src[2], src[0]));
    let (v1, v2) = (sub(dst[1], dst[0]), sub(dst[2], dst[0]));
    let det = cross(u1, u2);
    if det.abs() <= PARALLEL_TOLERANCE * length(u1) * length(u2) {
        return None;
    }
    // [v1 v2] times the inverse of [u1 u2], both with the vectors as columns
    let a = (v1.0 * u2.1 - v2.0 * u1.1) / det;
    let c = (v2.0 * u1.0 - v1.0 * u2.0) / det;
    let b = (v1.1 * u2.1 - v2.1 * u1.1) / det;
    let d = (v2.1 * u1.0 - v1.1 * u2.0) / det;
    let (sx, sy) = (src[0].x as f64, src[0].y as f64);
    let e = dst[0].x as f64 - (a * sx + c * sy);
    let f = dst[0].y as f64 - (b * sx + d * sy);
    Some([a, b, c, d, e, f].map(|v| v as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec2D, b: Vec2D) -> bool {
        (a - b).norm() < 1e-3
    }

    fn tilted() -> Quad {
        [
            Vec2D::new(110.0, 40.0),
            Vec2D::new(300.0, 10.0),
            Vec2D::new(320.0, 250.0),
            Vec2D::new(90.0, 200.0),
        ]
    }

    const TL: Vec2D = Vec2D { x: 10.0, y: 20.0 };
    const BR: Vec2D = Vec2D { x: 210.0, y: 120.0 };

    #[test]
    fn maps_rect_corners_onto_quad_corners() {
        let quad = tilted();
        let h = Homography::from_rect_to_quad(TL, BR, &quad).unwrap();
        let corners = [TL, Vec2D::new(BR.x, TL.y), BR, Vec2D::new(TL.x, BR.y)];
        for (corner, target) in corners.iter().zip(quad) {
            assert!(close(h.apply(*corner), target), "{corner} -> {target}");
        }
        for (mapped, target) in h.map_rect(TL, BR).iter().zip(quad) {
            assert!(close(*mapped, target));
        }
    }

    #[test]
    fn maps_rect_center_onto_intersection_of_quad_diagonals() {
        // a projective map keeps lines, so the crossing of the diagonals
        // stays their crossing, unlike the average of the corners
        let quad = tilted();
        let h = Homography::from_rect_to_quad(TL, BR, &quad).unwrap();

        let (p, r) = (quad[0], quad[2] - quad[0]);
        let (q, s) = (quad[1], quad[3] - quad[1]);
        let t = (q - p).cross(s) / r.cross(s);
        let crossing = p + r * t;

        assert!(close(h.apply((TL + BR) * 0.5), crossing));
        assert!(!close(
            crossing,
            quad.iter().fold(Vec2D::zero(), |a, b| a + *b) * 0.25
        ));
    }

    #[test]
    fn inverse_maps_points_back() {
        let h = Homography::from_rect_to_quad(TL, BR, &tilted()).unwrap();
        let inv = h.inverse().unwrap();
        for p in [
            Vec2D::new(50.0, 60.0),
            Vec2D::new(200.0, 30.0),
            Vec2D::new(-40.0, 300.0),
        ] {
            assert!(close(inv.apply(h.apply(p)), p), "{p}");
        }
    }

    #[test]
    fn same_rect_gives_identity_mapping() {
        let quad = [TL, Vec2D::new(BR.x, TL.y), BR, Vec2D::new(TL.x, BR.y)];
        let h = Homography::from_rect_to_quad(TL, BR, &quad).unwrap();
        for p in [Vec2D::new(3.0, 4.0), Vec2D::new(500.0, -7.0)] {
            assert!(close(h.apply(p), p));
            assert!(close(Homography::identity().apply(p), p));
        }
    }

    #[test]
    fn rejects_folded_concave_mirrored_and_collinear_quads() {
        let square = [
            Vec2D::new(0.0, 0.0),
            Vec2D::new(10.0, 0.0),
            Vec2D::new(10.0, 10.0),
            Vec2D::new(0.0, 10.0),
        ];
        assert!(is_convex_quad(&square));

        let mut folded = square;
        folded.swap(2, 3);
        let mut concave = square;
        concave[2] = Vec2D::new(3.0, 3.0);
        let mut mirrored = square;
        mirrored.reverse();
        let mut collinear = square;
        collinear[2] = Vec2D::new(10.0, 0.0);
        let mut degenerate_edge = square;
        degenerate_edge[1] = degenerate_edge[0];

        for quad in [folded, concave, mirrored, collinear, degenerate_edge] {
            assert!(!is_convex_quad(&quad), "{quad:?}");
            assert!(
                Homography::from_rect_to_quad(TL, BR, &quad).is_none(),
                "{quad:?}"
            );
        }
    }

    #[test]
    fn rejects_empty_rect() {
        assert!(Homography::from_rect_to_quad(TL, Vec2D::new(TL.x, BR.y), &tilted()).is_none());
    }

    #[test]
    fn then_translate_moves_every_image() {
        let h = Homography::from_rect_to_quad(TL, BR, &tilted()).unwrap();
        let delta = Vec2D::new(-30.0, 45.0);
        let moved = h.then_translate(delta);
        for p in [TL, BR, Vec2D::new(70.0, 90.0)] {
            assert!(close(moved.apply(p), h.apply(p) + delta), "{p}");
        }
    }

    #[test]
    fn denominator_changes_sign_across_the_horizon() {
        // a strong perspective, whose horizon is close to the rectangle
        let quad = [
            Vec2D::new(0.0, 0.0),
            Vec2D::new(100.0, 40.0),
            Vec2D::new(100.0, 60.0),
            Vec2D::new(0.0, 100.0),
        ];
        let (tl, br) = (Vec2D::zero(), Vec2D::new(100.0, 100.0));
        let h = Homography::from_rect_to_quad(tl, br, &quad).unwrap();
        for p in [tl, br, Vec2D::new(50.0, 50.0)] {
            assert!(h.denominator(p) > 0.0, "{p}");
        }
        // the short right side is the far one, so the denominator grows to the
        // right, and the points behind the viewer lie far to the left
        assert!(h.denominator(Vec2D::new(-1000.0, 50.0)) < 0.0);
    }

    #[test]
    fn depth_ratio_is_the_ratio_of_the_near_and_far_side() {
        let (tl, br) = (Vec2D::zero(), Vec2D::new(100.0, 100.0));
        let parallelogram = [
            Vec2D::new(0.0, 0.0),
            Vec2D::new(100.0, 20.0),
            Vec2D::new(130.0, 120.0),
            Vec2D::new(30.0, 100.0),
        ];
        let h = Homography::from_rect_to_quad(tl, br, &parallelogram).unwrap();
        assert!((h.depth_ratio(tl, br) - 1.0).abs() < 1e-9);

        // the right side is a quarter as long as the left one, so it is four
        // times as far away
        let trapezoid = [
            Vec2D::new(0.0, 0.0),
            Vec2D::new(100.0, 37.5),
            Vec2D::new(100.0, 62.5),
            Vec2D::new(0.0, 100.0),
        ];
        let h = Homography::from_rect_to_quad(tl, br, &trapezoid).unwrap();
        assert!((h.depth_ratio(tl, br) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn local_scale_is_the_largest_stretch() {
        let quad = [
            Vec2D::new(0.0, 0.0),
            Vec2D::new(300.0, 0.0),
            Vec2D::new(300.0, 50.0),
            Vec2D::new(0.0, 50.0),
        ];
        let (tl, br) = (Vec2D::zero(), Vec2D::new(100.0, 100.0));
        let h = Homography::from_rect_to_quad(tl, br, &quad).unwrap();
        assert!((h.local_scale(Vec2D::new(40.0, 70.0)) - 3.0).abs() < 1e-4);

        // the side that comes closer is stretched more than the far side
        let h = Homography::from_rect_to_quad(TL, BR, &tilted()).unwrap();
        let p = Vec2D::new(60.0, 60.0);
        let step = Vec2D::new(0.01, 0.0);
        let stretch = (h.apply(p + step) - h.apply(p)).norm() / step.norm();
        assert!(h.local_scale(p) >= stretch - 1e-3);
    }

    #[test]
    fn outset_moves_every_edge_by_the_distance() {
        let quad = tilted();
        let outset = outset_quad(&quad, 4.0);
        for i in 0..4 {
            let (a, b) = (quad[i], quad[(i + 1) % 4]);
            let normal = Vec2D::new(b.y - a.y, a.x - b.x) / (b - a).norm();
            for p in [outset[i], outset[(i + 1) % 4]] {
                // with y pointing down, the left normal of a clockwise edge
                // points outwards
                assert!(((p - a).dot(normal) - 4.0).abs() < 1e-3, "edge {i}: {p}");
            }
        }

        let square = [
            Vec2D::new(0.0, 0.0),
            Vec2D::new(10.0, 0.0),
            Vec2D::new(10.0, 10.0),
            Vec2D::new(0.0, 10.0),
        ];
        assert!(close(outset_quad(&square, 1.0)[0], Vec2D::new(-1.0, -1.0)));
    }

    #[test]
    fn affine_maps_triangle_vertices() {
        let src = [
            Vec2D::new(0.0, 0.0),
            Vec2D::new(4.0, 1.0),
            Vec2D::new(1.0, 5.0),
        ];
        let dst = [
            Vec2D::new(10.0, 3.0),
            Vec2D::new(-2.0, 8.0),
            Vec2D::new(7.0, 20.0),
        ];
        let [a, b, c, d, e, f] = affine_from_triangles(&src, &dst).unwrap();
        for (s, t) in src.iter().zip(dst) {
            let mapped = Vec2D::new(a * s.x + c * s.y + e, b * s.x + d * s.y + f);
            assert!(close(mapped, t), "{s} -> {mapped}, want {t}");
        }

        let flat = [
            Vec2D::new(0.0, 0.0),
            Vec2D::new(1.0, 1.0),
            Vec2D::new(2.0, 2.0),
        ];
        assert!(affine_from_triangles(&flat, &dst).is_none());
    }
}
