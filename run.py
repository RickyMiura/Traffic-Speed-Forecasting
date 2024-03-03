import sys


if __name__ == '__main__':
    targets = sys.argv[1:]
    if len(targets) == 0:
        from graphs import Graph1_SingleEdge, Graph2_SingleEdge, Graph3_SingleEdge, Graph4_SingleEdge, Graph5_SingleEdge, Graph6_SingleEdge, Graph1_EdgeType, Graph2_EdgeType, Graph3_EdgeType, Graph4_EdgeType, Graph5_EdgeType, Graph6_EdgeType
    
    if 'Graph1_SingleEdge' in targets:
        from graphs import Graph1_SingleEdge
    
    if 'Graph2_SingleEdge' in targets:
        from graphs import Graph2_SingleEdge
    
    if 'Graph3_SingleEdge' in targets:
        from graphs import Graph3_SingleEdge
    
    if 'Graph4_SingleEdge' in targets:
        from graphs import Graph4_SingleEdge

    if 'Graph5_SingleEdge' in targets:
        from graphs import Graph5_SingleEdge
    
    if 'Graph6_SingleEdge' in targets:
        from graphs import Graph6_SingleEdge
    
    if 'Graph1_EdgeType' in targets:
        from graphs import Graph1_EdgeType

    if 'Graph2_EdgeType' in targets:
        from graphs import Graph2_EdgeType
    
    if 'Graph3_EdgeType' in targets:
        from graphs import Graph3_EdgeType

    if 'Graph4_EdgeType' in targets:
        from graphs import Graph4_EdgeType

    if 'Graph5_EdgeType' in targets:
        from graphs import Graph5_EdgeType
    
    if 'Graph6_EdgeType' in targets:
        from graphs import Graph6_EdgeType