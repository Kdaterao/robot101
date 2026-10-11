# SAM 2 Gripper Filtering for POV Clustering

## Goal and current status

Use SAM 2 to mask the gripper in POV/wrist footage, exclude points on the
gripper, and give nearby points greater weight during clustering. Keep
BootsTAPIR for the first implementation. Target-object segmentation and
TAPNext++ migration are deferred.

The Molmo2 connector completed a 2,000-step training run and was uploaded as
`so101_connector.pt` to both `kdaterao/so101-molmo2-4b-gripper` and
`kdaterao/so101-molmo2-4b-gripper2`. Its localization quality still needs
evaluation on held-out images. Training token accuracy does not establish
gripper point accuracy.

The preprocessor implements this first milestone with Molmo2, SAM 2, and
BootsTAPIR. See [the run guide](preprocessing.md#sam-2-pov-gripper-filtering)
for setup, mask corrections, and configuration. Held-out model evaluation and
GPU mask-quality/throughput validation remain required before a large run.

## POV gripper masks and clustering

1. **Select and prompt the gripper.** Ask Molmo for points on visible gripper
   body or fingers and use them as positive SAM 2 prompts. Keep these separate
   from the jaw-center contact point: the latter can fall in empty space and
   is not a reliable segmentation prompt. Do not use a fixed pixel offset to
   turn a jaw-center prediction into a body point.
2. **Propagate the mask.** Initialize SAM 2 for each stage's clustering clip
   and propagate the gripper mask through its tail window, including the
   existing 10-frame extension into the next subtask. Cover the whole stage
   when shorter than the tail window; truncate extensions at the episode end.
3. **Exclude gripper points.** Reject candidate endpoints inside the gripper
   mask at the selection endpoint. Exclude masked trajectory observations
   from motion, endpoint, and proximity statistics rather than treating them
   as valid stationary observations.
4. **Weight nearby eligible points.** Increase clustering weight for points
   outside but near the mask. Use a maximum weight of **2×**, tapering linearly
   to **1×** over **5% of the source image diagonal**. Make the maximum weight
   and distance fraction configurable. Apply weights to eligible observations
   only; proximity must never override inside-mask exclusion.
5. **Preserve point tracking.** Keep tail-window clustering, shared candidate
   identities and cross-demo selection, backward BootsTAPIR tracking across
   each full stage, and the existing transition overlap. The mask filters
   clustering candidates; it does not replace point trajectories.
6. **Review unresolved masks.** Queue missing or invalid prompts/masks for
   human correction on the exact affected image. Do not treat a missing mask
   as confirmed background or silently publish unfiltered grounding as valid.
   Resume affected filtering after correction.

For eligible observations, the proposed proximity weight is
`w(d) = 1 + (w_max - 1) * max(0, 1 - d / r)`, where `d` is distance to the
nearest gripper-mask pixel and `r = 0.05 * sqrt(width² + height²)` by default.
Use source-image coordinates consistently when mapping masks and tracks.

SAM 2 is **POV-only** in this milestone and segments the **gripper only**.
Do not restrict candidates to a target-object mask. Nearby background remains
possible; proximity is a ranking signal, not proof of object contact.

## Molmo prompts and evaluation

Evaluate jaw-center predictions separately from gripper-body/finger prompts.
The existing jaw-center fine-tune does not establish reliable body prompting
or segmentation. Include held-out wrist and third-person images, open and
closed jaws, varied poses, partial occlusion, and subtask-end frames.

Return points in source-image pixel coordinates. Batch requests for selected
POV mask-initialization frames and exact third-person subtask-end frames;
avoid grounding every video frame. The current adapter has no calibrated
confidence score, so use explicit point/mask validity checks and human review
rather than an assumed confidence threshold.

## Third-person fallback: exact subtask end

For each available third-person camera, discover object and gripper candidates
on the exact subtask-start frame and track them forward. Select the goal object
by proximity to the gripper at the subtask endpoint, using the configured
proximity threshold and ambiguity margin. If the gripper track is lost,
re-detect the gripper on that exact endpoint frame for association. Preserve
this start-frame discovery and forward-tracking path; SAM 2 applies only to POV.

When object grounding is unresolved, permit a gripper-point fallback only if
Molmo detects the gripper on the **exact last frame of that subtask**. This
replaces the earlier episode-end-only restriction and applies to intermediate
and final subtasks alike.

- Never substitute a gripper detection from a nearby frame or another camera.
- Seed the detected endpoint point and track it backward through its subtask
  using the existing third-person tracking settings, including configurable
  sparse tracking FPS. Preserve the existing output transition overlap.
- Record the source as `gripper_fallback`, keeping it distinct from object
  grounding. A gripper fallback is not a verified object contact label.
- If the endpoint image is missing, or neither object nor gripper grounding
  is valid, queue human correction and mark the stage unresolved.
- Never use gripper fallback for POV clustering. SAM 2 remains POV-only.

Save the camera, source episode and subtask, exact seed frame, point coordinates,
grounding provenance, and review status in sidecar metadata. Use the actual
last included subtask frame, respecting the pipeline's boundary convention.

## Outputs and scaling

Keep training videos clean. Save point tracks, masks with frame associations,
fallback provenance, and review status as sidecars; viewers may render overlays.
Retain existing point-sidecar fields while adding the new metadata.

- Batch Molmo image requests where supported.
- Run SAM 2 per selected POV stage clip, not separately for each candidate.
- Preserve existing BootsTAPIR batching and cross-demo feature reuse.
- Parallelize Hub downloads and CPU decoding independently of GPU inference.
- Measure mask inference, tracking, decoding, and total episode time separately,
  including CPU RAM, GPU VRAM, and GPU utilization.

## Validation and milestones

1. **Evaluate the connector.** Measure point error and inspect held-out POV and
   third-person predictions, including open jaws and occlusion. Assess body
   prompts independently of contact-point predictions.
2. **Validate masks with BootsTAPIR.** Inspect SAM 2 gripper masks across tail
   windows and the 10-frame next-subtask extension. Test short stages, final
   stages, missing frames, and failed prompts/masks.
3. **Check filtering and weighting.** Inside-mask endpoints must be rejected;
   masked observations must not enter clustering statistics. Outside points
   must receive the configured weights, with 2× near the boundary and 1× at
   or beyond the default radius. Missing masks must require review.
4. **Check third-person fallback.** Cover intermediate and final subtasks,
   missing endpoint images, failed gripper detections, and valid object points.
   Confirm exact-frame seeding, backward tracking, `gripper_fallback`
   provenance, and absence of POV fallback.
5. **Inspect saved data.** Confirm clean videos, preserved transition overlap,
   aligned mask/point sidecars, and explicit unresolved/review status. Compare
   cluster quality and processing cost against the current pipeline.
6. **Benchmark TAPNext++ later.** Begin tracker migration only after gripper
   filtering works. Compare identical clips and point sets for tracking quality
   and end-to-end throughput before changing the default backend.

## Future milestone: TAPNext++

Keep BootsTAPIR active throughout the gripper-filtering milestone. A subsequent
TAPNext++ adapter must preserve point/visibility shapes, coordinate transforms,
reverse-stage tracking, sidecars, and transition overlap.

TAPIR's shared appearance-feature bank cannot simply be passed to TAPNext++.
Design and validate an alternative for candidate identity alignment across
demos before replacing that functionality. Retain BootsTAPIR for comparisons;
reported tracker FPS on different hardware does not establish an end-to-end
speedup for this pipeline.

## Relevant implementation

- [Grounded preprocessing](../src/robot101/data/preprocess.py)
- [Molmo2 inference adapter](../src/robot101/perception/molmo2.py)
- [Point tracking](../src/robot101/perception/tracking.py)
- [Gripper point collection](../src/robot101/data/collect_points.py)
- [Gripper point labeling guide](molmo_gripper_point_data.md)
- [Molmo2 training guide](molmo2_training.md)
- [TapNet checkout](../tapnet/README.md)
