# SAM 2 Contact-Guided Point Filtering

## Goal

Use the gripper's interaction with an object in the POV/wrist camera to
automatically prompt SAM 2, then use the object mask to remove irrelevant point
tracks and weight areas during POV clustering. Keep point tracking and
clustering responsible for producing the final point trajectories.

SAM 2 applies only to POV/wrist footage in this phase. Third-person cameras use
Molmo's initial object grounding points and point tracking. A third-person
gripper location may serve as a grounding reference only on the episode's
final frame, as described below.

This is a proposed workflow. The current dataset preprocessor still uses
BootsTAPIR and Molmo2; this document does not describe an implemented change.

## Proposed workflow

1. **Find likely contact moments.** Use gripper state changes and subtask
   boundaries to select one or two frames around the approach/close event. Prefer
   the last frame before contact, when the object is less likely to be occluded.
2. **Estimate the gripper contact point.** Fine-tune Molmo2 to point to the
   midpoint between the inner jaw tips. Collect labels from wrist and
   third-person views so the model sees both camera types. The current label
   guide asks annotators to click the jaw center; make the contact-point
   definition explicit before extending that dataset.
3. **Prompt SAM 2 on the POV frame with the interaction.** Use a visible point
   on the object immediately beyond the estimated jaw contact point as the positive prompt.
   The point may need a small offset if the jaw hides the object. Propagate the
   selected object mask through the relevant POV subtask frames.
4. **Track point candidates.** Generate candidate point tracks on the POV
   (wrist) video, then retain candidates that lie inside or near the selected
   object mask around contact and remain consistent with the propagated mask
   while visible. Give higher priority to points near the contact location.
   Cull candidates that remain outside the target mask. Allow configurable
   boundary tolerance and brief tracking or segmentation misses.
5. **Cluster and save points.** Keep the existing shared-candidate clustering
   logic initially. Save the point tracks and mask metadata as sidecar data;
   keep the source videos clean.

The mask is an object-selection and filtering signal. It does not replace the
precise point trajectory, and SAM 2 does not semantically understand parts such
as “gripper finger.”

## Gripper endpoint data

The intended label is the contact location between the inner jaw tips, not an
arbitrary point on the gripper body. Include open, closed, approaching,
partially occluded, wrist-camera, and third-person examples. The model should
return a point in source-image pixel coordinates. Run it only on selected
POV contact frames and third-person episode-end frames, and batch requests
across episodes rather than grounding every video frame. Plan to fine-tune on
a few thousand labeled gripper images covering these views and conditions;
include final-frame third-person examples in the collection.

If confidence is low, check temporal consistency across the nearby sampled
frames. For POV frames, also check whether the proposed contact point has a
plausible nearby object mask. Route only unresolved cases to human review.

## Third-person gripper reference: episode end only

Molmo continues to provide task-conditioned object grounding points for the
third-person cameras. The fine-tuned gripper detector can provide an additional
reference on the exact final frame of each episode (`frame_index == n_frames - 1`)
for each available third-person camera. Use its contact-point location to help
rank or validate nearby object grounding candidates on that same frame.

This permission applies to the episode endpoint, not every subtask endpoint.
Earlier third-person subtasks use object grounding without gripper-based
association or a requirement to detect the gripper. SAM 2 remains POV-only.
Keep the gripper reference distinct from the selected object point: detecting
the gripper alone does not establish a valid object grounding label.

Use the exact endpoint image. If that camera frame or the gripper detection is
unavailable, mark the reference unresolved and retain any independently valid
object grounding. Do not substitute a gripper position from a nearby frame.
Record the reference camera, episode/frame index, coordinates, validity, and
whether it influenced object selection in the point sidecar.

During implementation, remove the current third-person stage logic's dependence
on gripper detection and gripper/object proximity at every stage. Gate the new
reference path explicitly on the final episode frame. Tracking from a selected
endpoint object point should stay within its relevant subtask.

## Selected tracker: TAPNext++

The active preprocessor currently creates `BootsTAPIR` in
`src/robot101/data/preprocess.py`; the wrapper lives in
`src/robot101/perception/tracking.py`. The TapNet checkout includes TAPNext and
TAPNext++ implementations. The intended replacement is **TAPNext++**. This
section is the implementation plan; the active preprocessor has not yet been
converted.

TAPNext++ is selected for its long-sequence and re-detection focus. Its authors
report up to 562 FPS for 256 query points on an H100, while TAPIR reports about
40 FPS online for 256 points at 256×256 on a modern GPU. These figures use
different hardware and settings, so they motivate a local benchmark but do not
predict the end-to-end speedup for this preprocessor.

### Conversion steps

1. Add a TAPNext++ adapter alongside `BootsTAPIR` in
   `src/robot101/perception/tracking.py`, matching the existing point and
   visibility tensor shapes used by preprocessing.
2. Load the official TAPNext++ checkpoint and preprocessing configuration from
   the TapNet checkout; document the checkpoint download and runtime
   dependencies.
3. Replace the preprocessing tracker's default with TAPNext++ while retaining
   BootsTAPIR as an explicit comparison option during the transition.
4. Preserve third-person Molmo grounding, POV SAM 2 filtering, saved sidecar schema, and
   clean-video output. Preserve the clustering objective and selection rules,
   while allowing tracker-specific query initialization to change.
5. Adapt endpoint-to-start tracking by reversing the stage frame sequence,
   seeding at the endpoint, and mapping returned tracks and visibility back to
   chronological frame order. Validate query-frame indexing, coordinate
   transforms, state resets, and masks against that reversed ordering.
6. Replace TAPIR's shared appearance-feature bank with TAPNext++ query points
   and recurrent states. Initialize candidate points in each episode's sampled
   tail frames, then test how to align candidate identities across demos;
   TAPIR's cached descriptors cannot simply be passed into TAPNext++.
7. Compare BootsTAPIR and TAPNext++ on the same episodes and sampled frames
   before removing the old backend or changing other pipeline behavior.

## Scaling plan

- Use gripper-state changes to select a small number of contact frames.
- Batch Molmo requests for POV contact frames and third-person episode-end
  references across episodes on the GPU.
- Run SAM 2 once per selected POV object/subtask and propagate its mask through
  sampled wrist frames; avoid running a separate segmentation request per candidate
  point.
- Batch candidate queries through the point tracker where supported.
- Keep Hugging Face frame downloads and CPU video decoding parallelized
  independently from GPU inference.
- Store masks and confidence/review metadata as sidecars rather than drawing
  overlays into training videos.

## Evaluation before replacing the current pipeline

Use the same representative episodes and frame samples for BootsTAPIR and
TAPNext++. Include short and long stages, gripper occlusion, object motion, and
multiple subtasks. Compare:

- tracker-only time and end-to-end time;
- peak CPU RAM and GPU VRAM, plus GPU utilization;
- point visibility and manual point-position error on a small labeled subset;
- fraction of POV candidates retained by the object mask;
- cluster consistency and the number of unresolved masks or contact points.

Check that third-person gripper references occur only at `n_frames - 1`,
including episodes with multiple subtasks. Cover missing endpoint camera
frames and failed gripper detections; earlier third-person stages must still
ground objects independently.

The conversion target is TAPNext++. Keep BootsTAPIR available until TAPNext++
preserves useful point tracks and improves end-to-end processing time. The
current pipeline has had low average GPU utilization, so tracker speed alone
may not determine overall throughput.

## Relevant implementation

- [Grounded preprocessing](../src/robot101/data/preprocess.py)
- [Molmo2 inference adapter](../src/robot101/perception/molmo2.py)
- [Gripper point collection](../src/robot101/data/collect_points.py)
- [Gripper point labeling guide](molmo_gripper_point_data.md)
- [TapNet checkout](../tapnet/README.md)
