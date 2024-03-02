import sys


if __name__ == '__main__':
    targets = sys.argv[1:]
    if len(targets) == 0:
        from graphs import Graph1_SingleEdge, Graph2_SingleEdge, Graph3_SingleEdge, Graph4_SingleEdge, Graph5_SingleEdge, Graph6_SingleEdge, Graph1_EdgeType, Graph2_EdgeType, Graph3_EdgeType, Graph4_EdgeType, Graph5_EdgeType, Graph6_EdgeType
    
    if 'Graph 1 Single Edge' in targets:
        from graphs import Graph1_SingleEdge
    
    if 'Graph 2 Single Edge' in targets:
        from graphs import Graph2_SingleEdge
    
    if 'Graph 3 Single Edge' in targets:
        from graphs import Graph3_SingleEdge
    
    if 'Graph 4 Single Edge' in targets:
        from graphs import Graph4_SingleEdge

    if 'Graph 5 Single Edge' in targets:
        from graphs import Graph5_SingleEdge
    
    if 'Graph 6 Single Edge' in targets:
        from graphs import Graph6_SingleEdge
    
    if 'Graph 1 Edge Type' in targets:
        from graphs import Graph1_EdgeType

    if 'Graph 2 Edge Type' in targets:
        from graphs import Graph2_EdgeType
    
    if 'Graph 3 Edge Type' in targets:
        from graphs import Graph3_EdgeType

    if 'Graph 4 Edge Type' in targets:
        from graphs import Graph4_EdgeType

    if 'Graph 5 Edge Type' in targets:
        from graphs import Graph5_EdgeType
    
    if 'Graph 6 Edge Type' in targets:
        from graphs import Graph6_EdgeType