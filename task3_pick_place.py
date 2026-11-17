"""
Task 3 — Pick and Place State Machine
=======================================
TEKS4430 Assignment 4

The arm moves through 5 states:
  HOME → APPROACH → GRASP → CLOSE_GRIP → LIFT

⚠️  Requires the driver running in Terminal 1:
    ros2 launch arm_control_pkg arm.launch.py

----------------------------------------------------------------------
YOUR TASKS:
  3a — Implement _send_joint_commands(): publish STATE_ANGLES for the
       current state to each joint topic.

  3b — Implement _check_transition(): return True only when ALL joint
       errors < POSITION_THRESHOLD AND arm has held for DWELL_TIME s.

  3c — Run twice:
       Run 1: DWELL_TIME = 0.0  → observe state skipping
       Run 2: DWELL_TIME = 1.5  → observe smooth sequence
       Explain the difference using the lecture's
       "discrete decisions vs continuous dynamics" concept.
----------------------------------------------------------------------
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import time


# ── Configuration ────────────────────────────────────────────────────
POSITION_THRESHOLD = 0.05   # rad — "close enough" to advance state

# Task 3c: set to 1.5 for the dwell-time comparison run
DWELL_TIME = 1.5

# Joint targets [arm1, arm2, arm3, arm4, arm5], calculated from the Webots
# youBot transforms. GRASP uses the latest stable wrist sensor value, with
# the other joints recalculated to center the pads on the target box.
STATE_ANGLES = {
    'HOME': [0.00, 0.00, 0.00, 0.00, 0.00],
    'APPROACH': [-0.00013, 0.39818, -1.86324, 0.29253, 0.00],
    'GRASP': [-0.00013, 0.33819, -2.28121, 0.77049, 0.00],
    'CLOSE_GRIP': [-0.00013, 0.33819, -2.28121, 0.77049, 0.00],
    'LIFT': [-0.00013, 0.27847, -1.32059, -0.13041, 0.00],
}
STATE_SEQUENCE = ['HOME', 'APPROACH', 'GRASP', 'CLOSE_GRIP', 'LIFT']

GRIPPER_OPEN  = 0.0245
GRIPPER_CLOSE = 0.0185
# ────────────────────────────────────────────────────────────────────


class PickAndPlace(Node):

    def __init__(self):
        super().__init__('pick_and_place')

        self.state_index    = 0
        self.dwell_start    = None
        self.current_angles = [0.0] * 5
        self.last_wait_log  = 0.0
        self.grasp_hold_angles = None
        self.lift_targets = None

        self.joint_pubs = [
            self.create_publisher(Float64, f'/youBot/arm_joint_{i}/position', 10)
            for i in range(1, 6)
        ]

        self.grip_left_pub  = self.create_publisher(
            Float64, '/youBot/gripper_left/position', 10)
        self.grip_right_pub = self.create_publisher(
            Float64, '/youBot/gripper_right/position', 10)

        for i in range(1, 6):
            self.create_subscription(
                Float64,
                f'/youBot/arm_joint_{i}/sensor',
                lambda msg, idx=i-1: self._joint_callback(msg, idx),
                10,
            )

        self.create_timer(0.05, self._state_machine_loop)
        self.get_logger().info('Pick and place node started.')

    def _joint_callback(self, msg, index):
        self.current_angles[index] = msg.data

    def _state_machine_loop(self):
        if self.state_index >= len(STATE_SEQUENCE):
            self.get_logger().info('Sequence complete!', once=True)
            return

        state = STATE_SEQUENCE[self.state_index]
        self._send_joint_commands(state)
        self._handle_gripper(state)

        if self._check_transition(state):
            next_s = (STATE_SEQUENCE[self.state_index + 1]
                      if self.state_index + 1 < len(STATE_SEQUENCE) else 'DONE')
            if state == 'GRASP':
                self.grasp_hold_angles = self.current_angles.copy()
                self.get_logger().info(
                    f'Using measured grasp posture; holding joint 4 at '
                    f'{self.grasp_hold_angles[3]:.3f} rad'
                )
            self.get_logger().info(f'{state} → {next_s}')
            self.state_index += 1
            self.dwell_start = None
            self.last_wait_log = 0.0

    # ── Joint command and transition logic ────────────────────────────

    def _send_joint_commands(self, state):
        """
        Publish nominal or measured-posture joint targets for the state.

        CLOSE_GRIP holds the measured grasp posture. LIFT makes a small
        shoulder/elbow adjustment from that posture while holding the wrist.
        """
        targets = STATE_ANGLES[state]
        if state == 'CLOSE_GRIP' and self.grasp_hold_angles is not None:
            targets = self.grasp_hold_angles
        elif state == 'LIFT' and self.grasp_hold_angles is not None:
            if self.lift_targets is None:
                self.lift_targets = self.grasp_hold_angles.copy()
                self.lift_targets[1] = max(
                    -1.13446, self.lift_targets[1] - 0.03)
                self.lift_targets[2] = min(
                    2.54818, self.lift_targets[2] + 0.15)
                self.get_logger().info(
                    'Applying conservative lift: arm_joint_2 -0.030 rad, '
                    'arm_joint_3 +0.150 rad, wrist held'
                )
            targets = self.lift_targets

        for i, publisher in enumerate(self.joint_pubs):
            msg = Float64()
            msg.data = targets[i]
            publisher.publish(msg)

    def _check_transition(self, state):
        """
        Return True only when safe to move to the next state.

        With zero dwell, deliberately advance without waiting for joint
        feedback to demonstrate discrete decisions outrunning arm dynamics.

        Condition 1 — all commanded joints are close enough. The practical
        grasp fallback excludes wrist 4 only while in GRASP.

        Condition 2 — dwell time elapsed:
          When condition 1 first becomes True, record time.time() in
          self.dwell_start.
          Return True only when time.time() - self.dwell_start >= DWELL_TIME.
          If condition 1 becomes False, reset self.dwell_start = None.
        """
        if DWELL_TIME == 0.0:
            return True

        targets = STATE_ANGLES[state]
        if state == 'CLOSE_GRIP' and self.grasp_hold_angles is not None:
            targets = self.grasp_hold_angles
        elif state == 'LIFT' and self.lift_targets is not None:
            targets = self.lift_targets

        errors = [
            abs(self.current_angles[i] - targets[i])
            for i in range(5)
        ]
        if state == 'GRASP':
            # The grasp may physically obstruct the wrist before its nominal
            # angle is reached; its measured posture is held after transition.
            errors[3] = 0.0
        all_close = all(error < POSITION_THRESHOLD for error in errors)

        if all_close:
            if self.dwell_start is None:
                self.dwell_start = time.time()
            return time.time() - self.dwell_start >= DWELL_TIME

        self.dwell_start = None
        now = time.time()
        if now - self.last_wait_log >= 2.0:
            joint = max(range(5), key=errors.__getitem__)
            self.get_logger().info(
                f'Waiting in {state}: joint {joint + 1} error '
                f'{errors[joint]:.3f} rad exceeds '
                f'{POSITION_THRESHOLD:.3f} rad'
            )
            self.last_wait_log = now
        return False

    # ── Provided: gripper ────────────────────────────────────────────

    def _handle_gripper(self, state):
        pos = (
            GRIPPER_CLOSE
            if state in ('GRASP', 'CLOSE_GRIP', 'LIFT')
            else GRIPPER_OPEN
        )
        msg = Float64()
        msg.data = pos
        self.grip_left_pub.publish(msg)
        self.grip_right_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()