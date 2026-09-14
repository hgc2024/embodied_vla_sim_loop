// Mirrors the JSON messages dashboard/server.py broadcasts over /ws.
// Note this is a *dashboard-only* wire format (plain JSON) -- distinct from
// proto/schema.proto, which is the sim<->policy research wire format and is
// never exposed to the browser directly.

export interface FrameMessage {
  type: "frame";
  frame_id: number;
  timestamp_us: number;
  task_instruction: string;
  joint_positions: number[];
  joint_velocities: number[];
  rgb_jpeg: string; // base64-encoded JPEG
  depth_jpeg: string; // base64-encoded JPEG (colorized depth)
}

// Third-person view -- see mujoco_env.py's render_overview() docstring for
// why it's a separate message from FrameMessage rather than another field
// on it.
export interface OverviewMessage {
  type: "overview";
  frame_id: number;
  overview_jpeg: string; // base64-encoded JPEG
}

export type ActionMode = "predicted" | "fallback" | "scripted";

export interface StatusMessage {
  type: "status";
  timestamp_us: number;
  control_hz: number;
  frame_id: number;
  action_age_ms: number | null;
  mode: ActionMode;
  fallback_steps: number;
  paused: boolean;
  domain_randomization_enabled: boolean;
}

export type DashboardMessage = FrameMessage | OverviewMessage | StatusMessage;
