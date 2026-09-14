import type { FrameMessage, OverviewMessage } from "../types";

interface Props {
  frame: FrameMessage | null;
  overview: OverviewMessage | null;
}

export function CameraPanel({ frame, overview }: Props) {
  return (
    <section className="panel camera-panel">
      <h2>Scene</h2>
      <p className="task-instruction">
        {frame === null ? " " : `"${frame.task_instruction}"`}
      </p>

      {overview === null ? (
        <p className="muted">Waiting for the first frame…</p>
      ) : (
        <img
          className="overview-image"
          src={`data:image/jpeg;base64,${overview.overview_jpeg}`}
          alt="Third-person view of the arm and workspace"
        />
      )}

      <div className="wrist-cam-row">
        <p className="muted small hint" title="Mounted on the arm itself, not watching it from across the room -- so it's often an extreme close-up of whatever surface is nearest. That's normal.">
          Wrist camera (what the arm itself "sees")
        </p>
        {frame === null ? (
          <p className="muted small">Waiting…</p>
        ) : (
          <div className="camera-grid">
            <figure>
              <img src={`data:image/jpeg;base64,${frame.rgb_jpeg}`} alt="Wrist RGB camera feed" />
              <figcaption>RGB</figcaption>
            </figure>
            <figure>
              <img src={`data:image/jpeg;base64,${frame.depth_jpeg}`} alt="Wrist depth camera feed" />
              <figcaption
                className="hint"
                title="Same view, colored by distance instead of color: purple is near, red is far (up to 3m)."
              >
                Depth
              </figcaption>
            </figure>
          </div>
        )}
      </div>

      {frame !== null && <p className="muted small">frame #{frame.frame_id}</p>}
    </section>
  );
}
