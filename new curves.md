
# Technical Report: Mathematical Modeling of the Armscye (Armhole) Curve
---
## 1. Overall Shape and Geometric Character
### The Two-Zone Nature of the Armhole
Every armhole drafting tradition recognizes two geometrically distinct zones, which correspond to your O→P/BB (lower) and P/BB→shoulder (upper) distinction:
**Lower zone (underarm to chest-width/back-width line):** This is the strongly concave, more tightly curved section. The curve sweeps aggressively inward from the underarm point, reaches maximum inward penetration, then reverses toward the front/back width line. This is the zone with the highest curvature, and it is where the armhole is concave when viewed from outside the garment.
**Upper zone (chest-width/back-width line to shoulder tip):** This section has shallower, more complex curvature. On the front it is nearly straight to gently concave; on the back it has a slight convexity (outward bow) to accommodate the shoulder blade. The two zones together mean the **full armhole has an inflection point** — curvature sign changes — at the cross-front / cross-back line (the "chest width" or "across back" line). This inflection point is the most geometrically significant feature of the armhole.
### Concave vs. Convex Direction
Viewed from the outside (armhole opening facing you), the **back armhole** has:
- A concave lower section (curves away from you)
- A slightly convex upper section (bows toward you at the shoulder blade area)
The **front armhole** tends to be:
- Concave throughout, with the lower section more deeply concave than the upper
- The front upper section is more nearly flat/linear than the back
Multiple sources, including forum discussions and pattern books, explicitly state: "the front armhole is more scooped out and rounded, while the back is closer to vertical" (in the lower section).
### The Inflection Point (Cross-Front/Cross-Back)
This is one of the most consistent findings across all sources. The inflection point — where the curve transitions from one type of concavity to another — falls at the **chest-width line (front)** and the **across-back line (back)**. One forum contributor's phrase: "Various pattern drafting systems use this line to indicate where the edge of the curve should touch — the Bunka chest width line, the M&S front pitch line, the Myoungok interscye line." The tangent to the armhole at this inflection point is **vertical** (parallel to the grain/center front line). This is the geometrically most important constraint: the curve at the cross-front/back point has a vertical tangent, meaning its derivative with respect to the horizontal axis is infinite at that moment.
---
## 2. Traditional Textbook Construction Methods
### The Rectangle/Square Grid Method (Universal)
Almost every flat-pattern system (Aldrich, Beazley & Bond, Mueller & Sohn, Bunka, EMKO) starts by constructing a reference rectangle whose corners define the extremes of the armhole:
- **One corner** = underarm / side seam point (O)
- **Adjacent corner** = the vertical grid line at the armhole width (across-back or across-front measurement)
- The **armhole depth** (typically bust/4 + ease, or a directly measured scye depth) defines the horizontal extent
This rectangle is the bounding box of the armhole curve. The curve then passes through the interior of this box.
### Diagonal Guide Points (The "45-degree" Method)
This is nearly universal. From the corner of the reference rectangle (the "square" formed at the armhole width / armhole depth intersection):
- A diagonal line is drawn at **45 degrees** inward (toward the center of the garment)
- A point is marked on this diagonal at a fixed distance from the corner, typically **1.5 cm to 2.5 cm** (some sources say 1" for front, ¾" for back)
- This diagonal point is the key **pass-through constraint** for the curve in the lower section
In JBlockCreator's Ahmed bodice implementation, these are computed explicitly:
```
Vector2D apex1 = new Vector2D(point15.getX() - 1.414, point6.getY() + 1.414);  // sqrt(2) offset = 45°, 2cm
Vector2D apex2 = new Vector2D(point16.getX() + 1.7678, point6_2.getY() + 1.7678);  // 2.5cm
```
where 1.414 = √2 · 1.0 and 1.7678 = √2 · 1.25, confirming the 45-degree diagonal at the stated distances.
### The Midpoint / Notch Point
For the upper armhole section, most systems locate an intermediate point:
**Back:** The "across back" midpoint (cross-back level). This point is located at:
- Height: `armhole_depth / 2 + 2.0 cm` (Beazley & Bond: `Arb_BackArmholeTouchX = f_ArmholeDepth/2.0 + 2.0`)
- Width: `i_AcrossBack / 2.0` from center back
**Front:** A similar point at the front chest width line. JBlockCreator computes:
```java
Vector2D touchFront = new Vector2D(
    frontShoulderLineX + (2.0/3.0)*(f_ArmholeDepth - frontShoulderLineX),
    touchBack.getY() + Arb_WidthArmhole);
```
The front touch point is at 2/3 of the way along the armhole depth from the shoulder point.
The Modeliste Creative tutorial uses `(armhole girth + 2.5 cm) / 6` and `(armhole girth + 2.5 cm) / 4` to position intermediate points B and C vertically, with a 0.5 cm offset applied to C before drawing arcs. These fractions vary slightly by system but the 1/4 to 1/3 of armhole depth range is typical.
### French Curve vs. Compass vs. Spline
Traditional manual drafting uses a **French curve** template (a physical tool containing multiple curvature sections). The drafter selects the appropriate section of the French curve that passes through the three or four construction points simultaneously. The result is not mathematically prescribed beyond "passes through these points and meets the side seam squarely."
The EMKO (Soviet) system replaces French curves with **multi-arc construction**: the armhole is built from 4 circular arc segments, each computed by finding perpendicular bisectors. The forum post describes: "you make tangents and points on them, then draw perpendiculars and see where they cross, use the distance from points on curves to crossing points to find arc centers." Arc junctions are required to be tangent-continuous (G1).
---
## 3. Mathematical Representations in CAD/Software
### CAD Industry Standard: Cubic Bezier / Cubic Spline
Industrial garment CAD systems (Gerber AccuMark, Lectra Modaris) represent all curves as **NURBS or cubic Bezier segments**, which is the universal format in DXF/AAMA interchange. The Spiro/libspiro library (used by Inkscape and others) converts clothoid-based control points to Bezier output. No proprietary specification of armhole curves from Lectra or Gerber is public, but the DXF output of these systems stores curves as cubic Bezier segments.
### JBlockCreator (Open Source Java): Cubic Polynomial Spline with Prescribed Tangents
This is the most technically detailed open-source implementation found. The core method `addDirectedCubicSpline` fits a **cubic polynomial y = ax³ + bx² + cx + d** in a local reference frame, constrained by:
- 2 endpoint positions
- 2 endpoint gradient (tangent direction) conditions
The system builds a 4×4 linear system and solves it for coefficients [a, b, c, d]. This is equivalent to a Hermite cubic spline (C1 continuity across segments).
**Armhole-specific tangent conditions** (from `BodicePattern.java`):
The full armhole is split into **4 cubic segments** with prescribed tangent directions at each junction:
| Segment | From | To | Start tangent | End tangent |
|---------|------|----|--------------|------------|
| a (front upper) | Shoulder tip | Front touch point | `continuation of shoulder seam` | `(1.0, 0.0)` = horizontal |
| b (front lower) | Front touch point | Underarm | `(1.0, 0.0)` = horizontal | `(0.0, -1.0)` = vertical downward |
| c (back lower) | Underarm | Back touch point | `(0.0, -1.0)` = vertical | `(-1.0, 0.0)` = horizontal |
| d (back upper) | Back touch point | Shoulder tip | `(-1.0, 0.0)` = horizontal | `continuation of shoulder seam` |
```java
// Front shoulder → front touch: ends horizontal
Vector2D dirEnd = new Vector2D(1.0, 0.0);
// Front touch → underarm: starts horizontal, ends vertical
Vector2D dirStart = new Vector2D(1.0, 0.0);
Vector2D dirEnd = new Vector2D(0.0, -1.0);
// Underarm → back touch: starts vertical, ends horizontal
dirStart = dirEnd;  // (0.0, -1.0) vertical
dirEnd = new Vector2D(-1.0, 0.0);  // horizontal toward CB
```
The `addDirectedCubicSplineWithApexTangent` variant adds an intermediate keypoint at the apex of the lower curve:
- The apex is computed as the corner of the reference rectangle minus a diagonal offset
- The tangent at the apex is set perpendicular to the diagonal (i.e., at 45 degrees)
- This splits each lower section into two sub-cubics meeting at the apex with C1 continuity
### Seamly2D / Valentina (Open Source Qt/C++): Bezier Curves with the 0.551915 Factor
Seamly2D uses **cubic Bezier curves** where control point lengths are specified by formula. The canonical formula for a smooth circle-approximating Bezier is `handle_length = chord_length × 0.551915`. This constant (≈ 4(√2-1)/3) is the well-known Bezier circle approximation factor. Control point angles are derived from the adjacent seam lines:
- Perpendicular to the side seam → angle = 90° or 270°
- Parallel to the shoulder seam → angle derived from shoulder angle formula
- Adjustments of ±10–15° are used to eliminate unwanted S-curves
### GarmentCode (Research, Python/PyPI): Cubic Bezier in Local Edge Coordinates
The GarmentCode system (Korosteleva & Hilliges, SIGGRAPH 2023) uses **cubic Bezier edges** defined in a local coordinate system where:
- The straight segment between edge endpoints = unit horizontal axis
- Left perpendicular = vertical axis
- Both control points can be on one side (for convex/concave curves) or straddling (for S-curves)
For sleeve openings (the bodice side of the armhole), the paper states: "the projecting shape for a bodice is defined as a cubic Bezier curve with both control points on one side of the edge." The tangent directions at endpoints are constrained during optimization to ensure smooth panel junctions. The system optimizes control point positions to match desired tangent directions while preserving curve arc length.
### Academic Papers: Double-Ellipse Model
The Emerald paper "The mathematical model of men's garment prototype" (International Journal of Clothing Science and Technology, 2022) represents the armhole arc as a **double-ellipse superposition**: two elliptic arcs with shared or overlapping parameters that together trace the armhole shape. The front and back armhole are modeled as separate quarter-ellipses (1/4 of an ellipse each). The MDPI 2023 paper "Automatic Garment-Pattern-Generation System Based on Parameterized Design" similarly references front/back quarter-ellipse components as curve fitting targets.
The ellipse model works as follows: given armhole depth D and armhole width W, a quarter-ellipse with semi-axes a = W and b = D fits the lower portion. The perimeter of this quarter ellipse can be approximated via Ramanujan's formula (used in knitting calculations to match sleeve cap height to armhole perimeter).
---
## 4. Front Armhole: Tangent Directions
### At the Underarm Point (Point O, bottom of armhole)
**Tangent direction: horizontal** (perpendicular to the side seam, which runs vertically).
This is enforced by the "square corners" rule: where any curved seam meets a straight seam, they must meet at 90 degrees. Since the side seam is vertical, the armhole curve must leave O horizontally. JBlockCreator makes this explicit: `dirEnd = new Vector2D(0.0, -1.0)` for the segment arriving at the underarm (pointing straight down = arriving from above horizontally when the coordinate system is considered).
The "square corners" principle is stated by Sempstress.org: "where a curved edge (armscye) meets a straight edge (side seam), they must intersect perpendicularly." The mathematical consequence: the tangent to the armhole at O is horizontal (zero slope with respect to the vertical grain-line axis).
### At the Chest-Width Point (Point P / inflection point)
**Tangent direction: vertical** (parallel to the grain/center-front line).
This is the inflection point. The armhole curve reaches maximum lateral width here. JBlockCreator: `dirStart = new Vector2D(1.0, 0.0)` for the segment departing from the front touch point (horizontal in the code's coordinate system, which corresponds to vertical in the garment's grain-aligned frame).
### At the Shoulder Tip (top end)
**Tangent direction: continuation of the shoulder seam line**.
The armhole must meet the shoulder seam smoothly. Since the shoulder seam runs at the shoulder slope angle (typically 22–30 degrees from horizontal in most systems), the armhole curve arrives at the shoulder tip along this same direction. JBlockCreator: `new Vector2D(startPt.subtract(preStartPt))` = direction inherited from the shoulder dart edge.
---
## 5. Back Armhole: Tangent Directions
### At the Underarm Point (bottom)
**Same as front: horizontal tangent** at O (perpendicular to the vertical side seam). The "square corners" rule applies identically. JBlockCreator segment c: `dirStart = new Vector2D(0.0, -1.0)` arriving vertically into the underarm = departing horizontally from it.
### At the Across-Back Point (Point BB / back inflection)
**Tangent direction: vertical** (parallel to grain/center-back line).
This is the back-side inflection point. JBlockCreator segment c ends and d begins here: `dirEnd = new Vector2D(-1.0, 0.0)` (horizontal toward CB in code coordinates = vertical in grain-aligned frame).
The back across-back line is positioned slightly higher than the front chest-width line. Beazley & Bond place it at `armhole_depth/2 + 2 cm` (vs. front at about `2/3 × armhole_depth`).
### The Back-Specific Convexity
The back upper armhole (BB → shoulder tip) has a gentle outward bow (convexity toward the outside) to clear the shoulder blade. This is why the back armhole in segment d uses `addDirectedCubicSplineWithApexTangent` with the apex at the back reference rectangle corner, offset diagonally. The front equivalent (segment a) uses a simple directed cubic because the front upper armhole is nearly linear or gently concave. The Bunka system's documentation notes that the back is "more triangular" and closer to anatomically correct shoulder blade shape, while Western systems tend toward a more elliptical back.
### At the Shoulder Tip (back)
**Tangent direction: continuation of back shoulder seam** (which has a steeper slope than the front, typically 5–7 degrees steeper due to back shoulder slope conventions).
JBlockCreator: `new Vector2D(-Arb_BackNeckRise - endPt.getX(), Arb_HalfBackNeckWidth - endPt.getY())` = direction from shoulder tip toward the neck back point.
---
## 6. Inflection Points: Summary
| Location | Significance | Tangent direction |
|----------|-------------|-------------------|
| Underarm (O) | Meets side seam at 90° | Horizontal (perpendicular to vertical side seam) |
| Cross-front line (front P) | Inflection; max lateral width | Vertical (parallel to grain) |
| Cross-back line (back BB) | Inflection; max lateral width | Vertical (parallel to grain) |
| Shoulder tip | Meets shoulder seam | Along shoulder seam angle |
The curve has no inflection point in the lower zone. The lower zone is purely concave (bowl-shaped inward). The inflection occurs at the cross-front/back line, after which the upper zone may be roughly linear, gently concave, or (back only) gently convex.
---
## 7. Published and Open Resources for Programmers
### Open Source Code
- **JBlockCreator** — `https://github.com/aharwood2/JBlockCreator` — Java. Contains complete Beazley & Bond and Ahmed bodice blocks with full cubic spline armhole implementation. The `Block.java` file contains `addDirectedCubicSpline` and `addDirectedCubicSplineWithApexTangent` methods. Armhole construction is in `/src/beazleybond/BodicePattern.java` (4-segment cubic, explicit tangent directions) and `/src/ahmed/BodicePattern.java` (quadratic Bezier + cubic hybrid).
- **Seamly2D** — `https://github.com/FashionFreedom/Seamly2D` — Qt/C++. Uses cubic Bezier curves with angle/length parameterization. The 0.551915 Bezier circle approximation factor is used for handle lengths.
- **Valentina** — `https://gitlab.com/smart-pattern/valentina` — Qt/C++. Similar curve approach to Seamly2D.
- **Patro** — `https://github.com/FabriceSalvaire/Patro` — Python. Supports cubic Bezier, B-spline, and conic sections for pattern geometry.
- **Tau Meta Tau Physica** — `https://www.taumeta.org/` — Cubic Bezier with 4 control points as Inkscape extension.
### Academic Papers
- "The mathematical model of men's garment prototype," *International Journal of Clothing Science and Technology*, Vol. 34 No. 6, 2022 (Emerald). Double-ellipse armhole model.
- "Design and Research of Automatic Garment-Pattern-Generation System Based on Parameterized Design," *Sustainability*, MDPI, 2023. Three curve-fitting algorithms including quarter-ellipse for armhole.
- "GarmentCode: Programming Parametric Sewing Patterns," Korosteleva & Hilliges, *ACM Transactions on Graphics*, 2023 — `https://arxiv.org/html/2306.03642`. Cubic Bezier with local edge coordinates and tangent optimization.
- "The Determination of Armhole Curve of Similar Pattern for Margin Structure" — `https://www.researchgate.net/publication/266647483` — characteristic point method (shoulder point, armpit point, chest-width point).
### Forum Technical Discussions
- Seamly Patterns Forum, "Another very mathematical way of making curves" — `https://forum.seamly.io/t/another-very-mathematical-way-of-making-curves/1820` — EMKO multi-arc method with sine-rule refinement.
- Seamly Patterns Forum, "Super Easy Way to do Armhole Curves" — `https://forum.seamly.io/t/super-easy-way-to-do-the-armhole-curves/10620` — Bezier control point formulas with angle and length parameters.
---
## 8. Implementation Recipe for a Programmer
Based on the synthesis of all sources, here is a concrete implementation approach:
**Coordinate convention:** x = depth direction (toward waist), y = width direction (toward CF or CB).
**Key points to compute first:**
1. `O` = underarm/side-seam point = `(armhole_depth, side_seam_y)`
2. `P_front` = front touch/inflection = `(armhole_depth × 0.67, front_width_y)` approximately
3. `BB_back` = back touch/inflection = `(armhole_depth / 2 + 2 cm, back_width_y)` approximately
4. `S_front` / `S_back` = shoulder tip points
**Diagonal guide points (for the lower section):**
- Front: `P_lower_front = (front_width_x + 1.414 cm, underarm_y + 1.414 cm)` (45° diagonal, 2 cm from corner)
- Back: `P_lower_back = (back_width_x + 1.414 cm, underarm_y + 1.414 cm)` (45° diagonal, 2 cm from corner — slightly less than front)
**Curve construction (cubic Hermite / cubic Bezier, 4 segments):**
```
Segment 1 (front upper): S_front → P_front
  tangent at S_front: direction of shoulder seam (outward)
  tangent at P_front: (0, 1) = vertical (pointing toward CF)
  angle at S_front: 90° from shoulder seam = straight ahead along armhole
Segment 2 (front lower): P_front → O
  tangent at P_front: (0, 1) = vertical
  tangent at O: (-1, 0) = horizontal (pointing toward CB, away from CF)
  pass through P_lower_front diagonal point
Segment 3 (back lower): O → BB_back
  tangent at O: (1, 0) = horizontal (pointing toward CB)
  tangent at BB_back: (0, 1) = vertical (pointing toward CB)
  pass through P_lower_back diagonal point
Segment 4 (back upper): BB_back → S_back
  tangent at BB_back: (0, 1) = vertical
  tangent at S_back: direction of back shoulder seam (inward toward neck)
```
**For each segment, solve the 4×4 Hermite system:** given `P(t0) = p0`, `P(t1) = p1`, `P'(t0) = d0`, `P'(t1) = d1`, solve for cubic `ax³ + bx² + cx + d`. This is the exact method used by JBlockCreator.
**Bezier equivalent:** For a Bezier representation, convert Hermite tangents to control points: `C1 = P0 + d0/3`, `C2 = P1 - d1/3` (for t ∈ [0,1]). The factor 1/3 comes from the chain rule for cubic Bezier parameterization.
**Handle length heuristic (Seamly2D approach):** Use `handle_length = distance(start, end) × 0.551915`. This replicates the 90° arc Bezier approximation and gives smooth, visually natural curves.
---
## Key Takeaways
1. The armhole is **not a single mathematical primitive** — it is a composite curve with an inflection point at the cross-front/cross-back line.
2. The most mathematically rigorous single primitive that fits each half is a **quarter-ellipse** (academic papers) or a **cubic Hermite spline** (CAD/software).
3. The three immutable **tangent conditions** are: horizontal at underarm, vertical at cross-front/cross-back, and along the shoulder seam at the shoulder tip.
4. The lower section uses a **diagonal guide point at 45°, 1.5–2.5 cm from the armhole-depth/armhole-width corner** of the bounding rectangle.
5. The front and back differ mainly in: (a) back has a slight **convexity in the upper section** (shoulder blade clearance), (b) the back inflection point is **higher** relative to armhole depth than the front, (c) the front lower section curves **more aggressively** than the back.
6. JBlockCreator at `/tmp/JBlockCreator/src/beazleybond/BodicePattern.java` contains the cleanest open-source reference implementation, using 4 cubic segments with explicit tangent directions encoded as direction vectors.
---
Sources:
- [JBlockCreator GitHub (Beazley & Bond bodice)](https://github.com/aharwood2/JBlockCreator)
- [GarmentCode: Programming Parametric Sewing Patterns (arXiv)](https://arxiv.org/html/2306.03642)
- [Seamly2D: Super Easy Way to do Armhole Curves (forum)](https://forum.seamly.io/t/super-easy-way-to-do-the-armhole-curves/10620)
- [Seamly2D: Another Very Mathematical Way of Making Curves (forum)](https://forum.seamly.io/t/another-very-mathematical-way-of-making-curves/1820)
- [Seamly2D: Confused on Curves](https://forum.seamly.io/t/confused-on-curves-need-help/5987)
- [The mathematical model of men's garment prototype (Emerald/IJCST)](https://www.emerald.com/ijcst/article-abstract/34/6/892/123446/The-mathematical-model-of-men-s-garment-prototype)
- [Design and Research of Automatic Garment-Pattern-Generation System (MDPI Sustainability)](https://www.mdpi.com/2071-1050/15/2/1268)
- [Why Square Corners? (Sempstress)](https://www.sempstress.org/2014/why-square-corners/)
- [Drafting a Basic Bodice Block Explained (Modeliste Creative)](https://modelistecreative.com/2019/02/07/drafting-a-basic-bodice-block-explained/)
- [How to Draft the Basic Bodice Pattern (Shapes of Fabric)](https://www.theshapesoffabric.com/2019/10/07/how-to-draft-the-basic-bodice-pattern/)
- [Pattern Cutting Part 4: Draft the Armhole (Angela Kane)](https://angelakane.com/sewing_patterns/pattern-cutting/pattern-cutting-4.php)
- [Dresspatternmaking: Understanding the Sleeve Part 2](https://dresspatternmaking.com/blog/understanding-the-sleeve-part-2)
- [The Determination of Armhole Curve of Similar Pattern (Atlantis/ResearchGate)](https://www.researchgate.net/publication/266647483_The_Determination_of_Armhole_Curve_of_Similar_Pattern_for_Margin_Structure_and_Its_Sleeves_Arrangement)
- [Subtleties in Drafting Sleeves (ikat bag)](https://www.ikatbag.com/2014/03/subtelties-in-drafting-sleeves.html)
- [Bunka Style Basic Blocks (Yuzu and Pear)](https://www.yuzuandpear.com/post/bunka-style-basic-blocks-part-ii)
- [Patro Python Pattern Library](https://fabricesalvaire.github.io/Patro/index.html)
- [Tau Meta Tau Physica Open Source Pattern Making](https://www.taumeta.org/)
- [How to Draft a Bodice Block (In the Folds)](https://inthefolds.com/blog/2016/2/22/how-to-draft-a-bodice-block)
- [Drafting and Making Female Basic Bodice Block (Textile Learner)](https://textilelearner.net/drafting-and-making-women-basic-bodice-block/)
agentId: aae04938bfba76a84 (use SendMessage with to: 'aae04938bfba76a84' to continue this agent)
<usage>total_tokens: 85827
tool_uses: 85
duration_ms: 558157</usage>

Here's what the research turned up — there are several distinct options to consider, and understanding why the current curves are wrong helps narrow the choice.

Why a single quadratic fails here
The armhole has an inflection point — the curvature reverses direction — at the cross-chest/cross-back line (your P for front, BB for back). A quadratic Bézier has no inflection points, so it can never correctly model the full O-to-shoulder arc as a single segment. The current code also has the wrong tangent directions: the armhole must leave O horizontally (the "square corners" rule: where a curve meets a straight seam, they intersect at 90°), but the current _arm_cp construction is using a vertical departure from O.

The three immutable tangent conditions (agreed on by every source)
Point	Tangent direction	Reason
O (underarm)	horizontal	90° rule: side seam is vertical
P / BB (chest/back width line)	vertical	inflection point; maximum lateral width
N / DD (shoulder tip)	along shoulder seam	smooth junction with shoulder
Option A — Fix the single quadratics (simplest)
Keep one quadratic per arc-section, but compute the CP as the intersection of the correct tangent lines:

Front O→P: horizontal at O, vertical at P → CP = (P[0], O[1]) — the corner of the bounding rectangle
Back BB→O: vertical at BB, horizontal at O → CP = (BB[0], O[1])
This is the 45° construction rectangle corner that hand-drafters use. Clean, no new segments. The P→N and BB→DD lines stay straight (or can be slightly adjusted manually). Limitation: the upper arc (P→N and BB→DD) will still be straight lines.

Option B — Curve the upper sections too (cubic Hermite, 4 segments)
JBlockCreator's Beazley & Bond implementation splits the armhole into four cubic Hermite splines with prescribed tangent directions at each endpoint. Each cubic is defined by two endpoints + two tangent vectors; the Bézier control points are then C1 = P0 + d0/3, C2 = P1 − d1/3.

Front upper  (N  → P ): tangent along shoulder seam at N  →  vertical at P
Front lower  (P  → O ): vertical at P  →  horizontal at O
Back lower   (O  → BB): horizontal at O  →  vertical at BB
Back upper   (BB → DD): vertical at BB  →  tangent along shoulder seam at DD

The back upper section also gets a slight outward bow at the shoulder-blade area (modeled by placing the apex of the arc at the back-reference-rectangle corner offset diagonally). This is the most geometrically correct option and fixes the straight-line P→N problem too.

Option C — Quarter-ellipse approximation (cubic Bézier)
For each lower section, model it as a quarter-ellipse with semi-axes matching the arc extents and approximate with a cubic Bézier using the canonical k = 0.5519 factor:

Semi-axes: rx = |P[0] − O[0]|, ry = |P[1] − O[1]|
CP1 = O + (0, ry × 0.5519) (leaving O vertically... wait — actually leaving O horizontally)
→ CP1 = O + (−rx × 0.5519, 0) for the horizontal departure
→ CP2 = P + (0, −ry × 0.5519) for the vertical arrival at P
This produces a visually smooth, naturally proportioned curve. Less flexible than Hermite but very clean and requires no solver.

Option D — EMKO multi-arc method (G1-continuous circular arcs)
The Soviet EMKO system builds the armhole from 4 tangent-continuous circular arcs. Each arc is defined by finding the perpendicular bisector of two consecutive construction points; the arc center is where the bisectors intersect. No Bézier at all — just geometry. Visually very close to a French curve. Significantly more complex to implement but produces the "classic tailoring" look.

Summary
Option	Segments	Inflection handled	Curves P→N / BB→DD	Complexity
A — fixed quadratics	2	✓ (split at inflection)	✗ (still straight)	minimal
B — 4×cubic Hermite	4	✓	✓	moderate
C — quarter-ellipse	2	✓	✗	low
D — EMKO circular arcs	4–6	✓	✓	high
