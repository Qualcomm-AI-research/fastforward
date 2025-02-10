import copy
import dataclasses
import itertools

from collections.abc import Iterator, Sequence

import pytest

from typing_extensions import Self, override

from fastforward.autoquant.cfg import _dominance, blocks


@pytest.fixture(
    scope="module",
    params=[
        # Diamond
        #
        # ┌──A──┐
        # ▼     ▼
        # B     C
        # └─►D◄─┘
        ("AB", "AC", "BD", "CD"),
        # If/Else
        #   ┌───A
        #   ▼   │
        # ┌─B─┐ │
        # ▼   ▼ │
        # C   D │
        # │   │ │
        # │   │ ▼
        # └───┴►E
        ("AB", "BC", "BD", "CE", "DE", "AE"),
        # Sequence
        # A──►B──►C──►D──►E──►F
        ("AB", "BC", "CD", "DE", "EF"),
    ],
)
def graph(request: pytest.FixtureRequest) -> "_Graph":
    """
    Create `_Graph` from edge list.
    """
    return _Graph.from_edges(request.param)


def test_immediate_dominators(graph: "_Graph") -> None:
    # GIVEN: A CFG
    cfg = graph.cfg()

    # WHEN: Immediate dominators on the CFG are inferred
    _dominance.set_immediate_dominators(cfg)

    # WHEN: Immediate dominators are inferred from the test graph
    dominators = graph.dominators()
    immediate_dominators = _idom_from_dom(dominators)
    post_dominators = graph.invert().dominators()
    immediate_post_dominators = _idom_from_dom(post_dominators)

    # THEN: the CFG immediate dominators must match the test graph's immediate
    # dominators
    for block in _iterate_test_blocks(cfg):
        if (idom := block.immediate_dominator) is not None:
            assert isinstance(idom, _TestBlock)
            assert idom.label == immediate_dominators[block.label]
        else:
            assert immediate_dominators[block.label] is None

        if (idom_post := block.immediate_post_dominator) is not None:
            assert isinstance(idom_post, _TestBlock)
            assert idom_post.label == immediate_post_dominators[block.label]
        else:
            assert immediate_post_dominators[block.label] is None


def test_dominance(graph: "_Graph") -> None:
    # GIVEN: A CFG
    cfg = graph.cfg()

    # WHEN: Immediate dominators on the CFG are inferred
    _dominance.set_immediate_dominators(cfg)

    # WHEN: dominators are inferred from the test graph
    dominators = graph.dominators()
    blocks = {block.label: block for block in _iterate_test_blocks(cfg)}

    # THEN: the domination relations on the CFG must match those of the test
    # graph
    for label_a, label_b in itertools.product(graph.nodes, graph.nodes):
        block_a, block_b = blocks[label_a], blocks[label_b]
        if label_a in dominators.get(label_b, set()):
            assert block_b.is_dominated_by(block_a)
            assert block_a.dominates(block_b)
        else:
            assert not block_b.is_dominated_by(block_a)
            assert not block_a.dominates(block_b)


@dataclasses.dataclass(eq=False)
class _TestBlock(blocks.Block):
    """
    Test `Block` that can have any number of children.

    This `Block` is used for tests since the domination module does not rely
    on a specific type of block. It can hold a variable number of children and
    can therefore be a placeholder of various `Blocks`.
    """

    children: list[blocks.Block | None]
    label: str

    @override
    def set_tail(self, tail: blocks.Block) -> None:
        for i, child in enumerate(self.children):
            if child is None:
                self.children[i] = tail
            else:
                child.set_tail(tail)

    @override
    def named_children(self) -> Iterator[tuple[str, blocks.Block]]:
        for i, child in enumerate(self.children):
            if child is not None:
                yield f"child_{i}", child


@dataclasses.dataclass
class _Graph:
    """
    Graph used for constructing CFG test cases.

    This is a basic graph implementation that can infer domination
    relationships between different nodes in the graph. By inverting the edges,
    it can be used to find post-domination relationships.
    """

    nodes: set[str]
    edges: dict[str, list[str]]
    node_entry: str
    node_exit: str

    @classmethod
    def from_edges(cls, raw_edges: Sequence[str | tuple[str, str]]) -> Self:
        """
        Construct a `_Graph` from a list of edges.

        Args:
            raw_edges: A sequence of pairs that represent a directed egde. A
                pair can either be string of two characters or a tuple of two
                strings.

        Returns:
            A `_Graph` from `raw_edges`.

        Raises:
            `ValueError` if the graph does not have a single entry or exit
            node.
        """
        nodes: set[str] = set()
        edges: dict[str, list[str]] = {}
        for source, target in raw_edges:  # type: ignore[misc]
            nodes.add(source)
            nodes.add(target)
            edges.setdefault(source, []).append(target)

        nodes_outgoing = set(edges.keys())
        nodes_incoming = set(e for targets in edges.values() for e in targets)

        nodes_entry = nodes_outgoing - nodes_incoming
        nodes_exit = nodes_incoming - nodes_outgoing

        if len(nodes_entry) != 1:
            raise ValueError(f"{cls.__name__} must have exactly 1 entry node")
        if len(nodes_exit) != 1:
            raise ValueError(f"{cls.__name__} must have exactly 1 exit node")

        return cls(nodes, edges, nodes_entry.pop(), nodes_exit.pop())

    def invert(self) -> Self:
        """
        Invert the graph, creating a new graph.

        Creates a new graph where all edges are reversed.

        Returns:
            A newly created `_Graph` with inverted edges.
        """
        new_edges: dict[str, list[str]] = {}
        for source, targets in self.edges.items():
            for target in targets:
                new_edges.setdefault(target, []).append(source)
        return dataclasses.replace(
            self, edges=new_edges, node_entry=self.node_exit, node_exit=self.node_entry
        )

    def dominators(self) -> dict[str, set[str]]:
        """
        Infer domination relatioship of nodes in graph.

        Returns:
            A mapping from node names to the names of its dominators.
        """
        doms = {node: copy.copy(self.nodes) for node in self.nodes}
        open_set = [self.node_entry]
        visited_set: set[str] = set()

        while open_set:
            path = open_set.pop()
            node_current = path[-1]

            doms[node_current] &= set(path)

            for child in self.edges.get(node_current, []):
                new_path = path + child
                if new_path not in visited_set:
                    open_set.append(new_path)

        return doms

    def cfg(self) -> blocks.Block:
        """
        Construct a CFG that follows the structure of this graph.

        Returns:
            A CFG created out of `_TestBlock`s that follows the same structure
            as this graph.
        """
        cfg_nodes = {name: _TestBlock(children=[], label=name) for name in self.nodes}
        for source, targets in self.edges.items():
            cfg_node = cfg_nodes[source]
            for target in targets:
                cfg_node.children.append(cfg_nodes[target])
        entry_node = cfg_nodes[self.node_entry]

        assert not cfg_nodes[self.node_exit].children
        assert sum(1 for _ in entry_node.blocks()) == len(self.nodes), (
            "The constructed CFG must have an equal number of nodes as the graph"
        )
        return entry_node


def _idom_from_dom(dominators: dict[str, set[str]]) -> dict[str, str | None]:
    """
    Given a mapping of dominators, infer immediate dominators.

    Args:
        dominators: A mapping from node names to the set of dominator names.

    Returns:
        A mapping from node names to the name of the immediate dominator or
        `None` if there is no immediate dominator.
    """
    idoms: dict[str, str | None] = {}
    for node, doms in dominators.items():
        candidates = doms - {node}
        for candidate in candidates:
            if dominators[candidate] == candidates:
                idoms[node] = candidate
                break
        else:
            idoms[node] = None

    return idoms


def _iterate_test_blocks(block: blocks.Block) -> Iterator[_TestBlock]:
    """
    Iteratate over blocks in `block`s subgraph.

    Assert that `block` is a `_TestBlock` and that all of the blocks in its subgraph
    are also `_TestBlock`s.

    Yields:
        `_TestBlocks`s in `block`s subgraph.
    """
    assert isinstance(block, _TestBlock)
    for child in block.blocks():
        assert isinstance(child, _TestBlock)
        yield child
