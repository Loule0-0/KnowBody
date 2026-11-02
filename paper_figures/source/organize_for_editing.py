"""Organize native paper graphics into named, human-sized PowerPoint groups.

Apply to the original flat Fig.2/Fig.3 scene, before saving. Leaf geometry,
text, crops and image bytes are retained. IDs refer to those fixed scenes.
"""
from collections import Counter

from pptx.enum.shapes import MSO_SHAPE_TYPE


def ids(first, last):
    return list(range(first, last + 1))


def group(name, *children):
    return name, list(children)


def organize(slide, plan):
    original = {s.shape_id: s for s in slide.shapes}
    if any(s.shape_type == MSO_SHAPE_TYPE.GROUP for s in original.values()):
        raise ValueError('Expected the original flat scene; do not regroup an edited deck')

    def leaves(nodes):
        for node in nodes:
            if isinstance(node, int):
                yield node
            else:
                yield from leaves(node[1])

    requested = list(leaves(plan))
    if set(requested) != set(original) or any(n != 1 for n in Counter(requested).values()):
        raise ValueError('Every original shape must belong to exactly one semantic group')

    def build(node):
        if isinstance(node, int):
            return original[node]
        name, children = node
        shape = slide.shapes.add_group_shape([build(child) for child in children])
        shape.name = name
        # Keep generic vector primitives identifiable inside their own named group.
        for index, child in enumerate(shape.shapes, 1):
            if child.shape_type != MSO_SHAPE_TYPE.GROUP and child.name.startswith('Editable '):
                if child.has_text_frame and child.text:
                    child.name = 'Text | ' + child.text[:100]
                else:
                    child.name = name + ' | part ' + str(index)
        return shape

    for node in plan:
        build(node)
    if len(slide.shapes) != len(plan):
        raise ValueError('Unexpected ungrouped shapes left at slide level')


def organize_fig2(slide):
    g = group
    plan = [
        g('01 | (a) Interaction worlds',
          g('Section heading', 2, 3), 4,
          g('Real examples',
            g('Pen operation | photo + border + caption', 5, 6, 11),
            g('Pouring | photo + border + caption', 7, 8, 12),
            g('Real badge', 9, 10)),
          g('Simulation examples', 13,
            g('Push | photo + border + caption', 14, 15, 24),
            g('Cover | photo + border + caption', 16, 17, 25),
            g('Stack | photo + border + caption', 18, 19, 26),
            g('Tool use | photo + border + caption', 20, 21, 27),
            g('Simulation badge', 22, 23)),
          g('Observed feedback strip', 28, 29)),
        g('02 | KnowBody execution loop',
          g('Action arrow + label', 30, 31),
          g('Observation arrow + label', 32, 33), 34),
        g('03 | Record evidence link', 35, 36),
        g('04 | (b) Experience-guided execution',
          g('Section heading', 37, 38), 39,
          g('Execution agent', 40, 41, 42,
            g('Task input card', 43, 44, g('Task document icon', *ids(49, 54)), 55),
            g('Brief input card', 45, 46, 56, 57, 58),
            g('Live-state input card', 47, 48, 59, 60),
            g('Action chunk', 61, 62, 63,
              g('Free-space steps + bracket + caption', *ids(64, 68), *ids(76, 78), 82),
              g('Contact steps + bracket + caption', *ids(69, 75), *ids(79, 81), 83))),
          g('Retrieve', 84, 85,
            g('Same-task process card', 86, 87, 88, 89, 90, 91, 92, 93),
            g('Body / interaction knowledge card', *ids(94, 99)), 100),
          g('Query / return link', 101, 102),
          g('Memory manager', 103, 104,
            g('Prepare context', 105, g('Preparation document icon', *ids(106, 111)), 112),
            113,
            g('Review / learn', 114, g('Review lens icon', 115, 116), 117)),
          g('Episode-start link', 118, 119),
          g('On-demand management link', 120, 121),
          g('Brief / evidence return link', 122, 123),
          g('Source checks and validated write',
            g('Checked shield icon', 124, 125, 126), 127, 128,
            g('Input to append link', 129, 130),
            g('Append / revise operation', 131, 132),
            g('Complete write connector', 133, 134)),
          g('Manager reads library link', 135, 136),
          g('Retrieval reads library link', 137, 138)),
        g('05 | (c) Structured experience library', 139,
          g('Section heading', 140, 141),
          g('Organization rule', 142, 143),
          g('E / C / K level headings',
            g('Raw evidence column heading', 144, 145),
            g('Conditional cases column heading', 146, 147),
            g('Reusable knowledge column heading', 148, 149), 150, 151),
          g('Body experience row',
            g('Body raw-evidence card', 152,
              g('Body domain label', 153, 154, 155),
              g('Marker photo + EEF / tip annotations', *ids(156, 165))),
            166,
            g('Body conditional-case card', *ids(167, 178)),
            179,
            g('Body reusable-knowledge card', 180,
              g('Functional point photo + annotations', *ids(181, 190)), 191, 192)),
          g('External interaction experience row',
            g('External raw-evidence card', 193,
              g('External domain label', 194, 195, 196),
              g('Key grasp photo', 197, 198),
              g('Insertion access photo', 199, 200)),
            201,
            g('External conditional-case card', 202, 203,
              g('Grasp layout photo + caption', 204, 205, 209),
              g('Insertion access photo + caption', 206, 207, 210), 208),
            211,
            g('External reusable-knowledge card', *ids(212, 219))),
          g('Episode history strip', 220,
            g('Episode history label', 221, 222),
            g('History frame 1', 223, 224), g('History frame 2', 225, 226),
            g('History frame 3', 227, 228), g('History frame 4', 229, 230),
            231, 232, 233, 234, g('Source document icon', *ids(235, 240)), 241),
          g('Knowledge-first / details-on-demand links', 242, 243, 244)),
    ]
    organize(slide, plan)


def organize_fig3(slide):
    g = group
    plan = [
        g('01 | Body relations',
          g('Body section heading', 2, 3, 4),
          g('(a) Motion-image response', 5, 8,
            g('Before | photo + marker + label', 9, 11, 13, 14),
            g('After | photo + marker + label', 10, 12, 15, 16),
            17, g('Motion response explanation', 18, 19)),
          g('(b) EEF reference and acting tip', 6, 20,
            g('Tool photo + EEF / tip annotations', *ids(21, 27)),
            g('Tool relation explanation', 28, 29)),
          g('(c) Computed path check', 7, 30, 31,
            g('Path schematic', 32, 33,
              g('Direct invalid path', 34, 35),
              g('Invalid-path cross', 36, 37),
              g('Feasible path + waypoints', *ids(38, 45))),
            g('Path-check explanation', 46, 47))),
        g('02 | External interactions',
          g('External section heading', 48, 49, 50),
          g('(d) Grasp layout and downstream access', 51, 53,
            g('Grasp layout | photo + annotations + caption', 54, 56, 57, 60, 61, 62, 68),
            g('Insertion access | photo + annotations + caption', 55, 58, 59, 63, 64, 65, 69),
            g('Complete transition arrow', 66, 67),
            g('Downstream access explanation', 70, 71),
            g('Demonstration-candidate label', 72, 73)),
          g('(e) Retention across lift and transfer', 52, 74,
            g('Lift retained | photo + marker + caption', 75, 76, 77, 90),
            g('Transfer retained | photo + marker + caption', 78, 79, 80, 91),
            g('Later loss | photo + marker + caption', 81, 82, 83, 84, 85, 92),
            g('Lift-to-transfer arrow', 86, 87),
            g('Transfer-to-loss arrow', 88, 89),
            g('Reusable retention knowledge note', *ids(93, 97)))),
    ]
    organize(slide, plan)
