"""Verified model-facing operating notes for the Cartesian ARX X5 surface.

These live beside ``pose`` rather than with the joint cheat-sheet because every
number they quote is one of that module's constants, formatted here instead of
typed a second time. A figure the prose and the tool bounds each spell out on
their own drifts the first time the surface is retuned.

What the tool description, the system message, or the state block already says
is left out rather than repeated in a second wording. The tool owns the action
mechanics and the bounds, the system message owns what the model receives and
what it may not ask for, and these notes own what the spaces cannot express:
frame, geometry, grasp semantics, and gripper polarity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.embodiment_docs import (
    MOUNTING_HEADER,
    format_mounting_notes,
)

from .pose import GRASP_REACH_M, JAW_DEPTH_M, TABLE_SURFACE_Z

#: Lowest grasp point a straight-down jaw can reach before the table stops it.
_TABLETOP_FLOOR_Z = TABLE_SURFACE_Z + JAW_DEPTH_M / 2

#: Grasp point at which the jaws close entirely above the table surface, and so
#: entirely above anything lying flat on it.
_TABLETOP_CLEAR_Z = TABLE_SURFACE_Z + JAW_DEPTH_M


ARX_X5_EEF_DOCS = f"""Two identical 6-DoF ARX X5 arms, prefixed left_ and right_, each with a
parallel-jaw gripper. World frame: +z is up, both bases face world +y (see the
mounting note below), and +x is the lateral axis with the left arm on the -x
side. The table surface is at z = {TABLE_SURFACE_Z:.3f}, and each grasp point reaches about
{GRASP_REACH_M:.2f} m from its own base.
Every position, commanded and reported, is the point between the jaws where an
object is held. To grasp something resting on the table, name that object's
own position: no tool offset is yours to add.
The jaws are {JAW_DEPTH_M * 1000:.0f} mm deep along the tool axis and the commanded point is their
centre, so a straight-down grasp reaches no lower than z = {_TABLETOP_FLOOR_Z:.4f} before the
table stops it. Anything flat lying on the table, such as a tile, a card or a
token, is therefore grasped at a grasp-point z of about {_TABLETOP_FLOOR_Z:.4f} rather than at
an estimated centre height: naming a lower z is safe and lands in the same
place, while a z above {_TABLETOP_CLEAR_Z:.2f} closes the jaws entirely above such an object.
Objects tall enough to stand above the table are grasped at their own height
as usual.
Gripper 0 is fully closed, 1 is fully open, and commanding 0 always closes as
far as the object allows.
Each wrist camera rides its own arm; camera geometry is observation dependent.
The current body projections are available through read_body and do not prove
object contact, clearance or retention. Objects carried by the jaws extend the
robot's effective geometry; their offsets and rigidity require evidence.
When required by the public task recipe, returning both arms within 0.15 m and
20 degrees of their episode-start poses is part of completion.
A target inside the advertised bounds may still be refused by the planner.
Such a refusal is a computational result under that start and configuration,
not proof of physical impossibility; neither arm moves on a rejected plan."""


def build_arx_x5_eef_docs(mount_poses: Mapping[str, Sequence[float]] | None = None) -> str:
    """Cartesian operating notes, with the same live mounting appendix as joint docs."""
    mounting = format_mounting_notes(mount_poses or {})
    if not mounting:
        return ARX_X5_EEF_DOCS
    return ARX_X5_EEF_DOCS + "\n\n" + mounting


def eef_docs_from_joint_docs(joint_docs: str) -> str:
    """Keep the live mounting appendix, replace the joint cheat-sheet."""
    _, found, rest = joint_docs.partition(MOUNTING_HEADER)
    if not found:
        return ARX_X5_EEF_DOCS
    return ARX_X5_EEF_DOCS + "\n\n" + MOUNTING_HEADER + rest
