import React, { useState, useEffect, useCallback } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './MemoryTreeVisualization.css';
import queuedFetch from '../utils/requestQueue';
import { useTranslation } from 'react-i18next';

const MemoryTreeVisualization = ({ 
  memoryType, 
  serverUrl = 'http://localhost:47283',
  getItemTitle = (item) => item.title || item.name || item.filename || item.summary,
  getItemDetails = (item) => ({ summary: item.summary, details: item.details })
}) => {
  const { t } = useTranslation();
  const [memoryItems, setMemoryItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [expandedCategories, setExpandedCategories] = useState(new Set());
  const [allNodes, setAllNodes] = useState([]); // Store all nodes
  const [allEdges, setAllEdges] = useState([]); // Store all edges
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    loadMemoryData();
  }, [serverUrl, memoryType]);

  const loadMemoryData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const endpointMap = {
        'episodic': '/memory/episodic',
        'procedural': '/memory/procedural', 
        'resource': '/memory/resources',
        'semantic': '/memory/semantic'
      };
      
      const endpoint = endpointMap[memoryType];
      if (!endpoint) {
        throw new Error(t('memory.states.treeError', { error: `Unknown memory type: ${memoryType}` }));
      }

      const response = await queuedFetch(`${serverUrl}${endpoint}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setMemoryItems(data);
      
      // Reset expanded categories when refreshing
      setExpandedCategories(new Set());
      
      // Build graph structure
      const { nodes: graphNodes, edges: graphEdges } = buildGraphFromItems(data);
      setAllNodes(graphNodes);
      setAllEdges(graphEdges);
      
      // Initially show only category nodes (collapsed state)
      updateVisibleNodes(graphNodes, graphEdges, new Set());
      
      setLoading(false);
    } catch (err) {
      setError(t('memory.states.treeError', { error: err.message }));
      setLoading(false);
    }
  };

  const buildGraphFromItems = (items) => {
    const nodes = [];
    const edges = [];
    const nodeMap = new Map();
    
    // Build hierarchy from tree paths
    const pathGroups = {};
    
    items.forEach(item => {
      const treePath = item.tree_path && item.tree_path.length > 0 ? item.tree_path : ['uncategorized'];
      const pathKey = treePath.join('/');
      
      if (!pathGroups[pathKey]) {
        pathGroups[pathKey] = {
          path: treePath,
          items: []
        };
      }
      pathGroups[pathKey].items.push(item);
    });

    // Calculate layout dimensions
    const levelHeight = 120;
    const nodeWidth = 180;
    const nodeSpacing = 40;
    
    // Build tree structure with proper hierarchical positioning
    const levelNodes = {}; // Track nodes at each level
    
    Object.values(pathGroups).forEach((group) => {
      let parentId = null;
      let currentPath = '';
      
      // Only create nodes for the path, excluding the final level (which will be memory items)
      group.path.slice(0, -1).forEach((segment, levelIndex) => {
        const nodePath = currentPath ? `${currentPath}/${segment}` : segment;
        
        if (!nodeMap.has(nodePath)) {
          // Initialize level if not exists
          if (!levelNodes[levelIndex]) {
            levelNodes[levelIndex] = [];
          }
          
          const node = {
            id: nodePath,
            type: 'default',
            data: { 
              label: segment,
              type: 'category',
              count: 0,
              items: [],
              path: group.path.slice(0, levelIndex + 1),
              isExpandable: false, // Will be set to true if it has memory items as children
            },
            position: { x: 0, y: 0 }, // Will be calculated later
            className: 'category-node',
          };
          
          nodes.push(node);
          nodeMap.set(nodePath, node);
          levelNodes[levelIndex].push(node);
          
          // Create edge from parent to this node
          if (parentId) {
            edges.push({
              id: `${parentId}-${nodePath}`,
              source: parentId,
              target: nodePath,
              type: 'smoothstep',
              animated: false,
            });
          }
        }
        
        parentId = nodePath;
        currentPath = nodePath;
      });
    });

    // Group memory items by their parent for horizontal positioning
    const memoryItemsByParent = {};
    Object.values(pathGroups).forEach((group) => {
      if (group.items.length > 0) {
        const parentPath = group.path.slice(0, -1).join('/');
        if (!memoryItemsByParent[parentPath]) {
          memoryItemsByParent[parentPath] = [];
        }
        
        group.items.forEach((item) => {
          memoryItemsByParent[parentPath].push({
            item,
            group
          });
        });
      }
    });

    // Mark category nodes as expandable if they have memory items as children
    Object.keys(memoryItemsByParent).forEach((parentPath) => {
      const parentNode = nodeMap.get(parentPath);
      if (parentNode) {
        parentNode.data.isExpandable = true;
        parentNode.data.childCount = memoryItemsByParent[parentPath].length;
        parentNode.data.originalLabel = parentNode.data.label; // Store original label
        // Update label to show count and expand/collapse indicator
        parentNode.data.label = `${parentNode.data.originalLabel} (${parentNode.data.childCount}) +`;
      }
    });

    // Now add individual memory items as bottom-level nodes
    const memoryItemLevel = Math.max(...Object.keys(levelNodes).map(Number)) + 1;
    levelNodes[memoryItemLevel] = [];
    
    Object.entries(memoryItemsByParent).forEach(([parentPath, items]) => {
      items.forEach((itemData, itemIndex) => {
        const itemId = `${itemData.group.path.join('/')}/item-${itemIndex}`;
        const itemNode = {
          id: itemId,
          type: 'output',
          data: {
            label: getItemTitle(itemData.item),
            type: 'memory-item',
            memoryItem: itemData.item,
            path: itemData.group.path,
            parentPath: parentPath
          },
          position: { x: 0, y: 0 }, // Will be calculated later
          className: 'memory-item-node',
        };
        
        nodes.push(itemNode);
        levelNodes[memoryItemLevel].push(itemNode);
        
        // Connect to parent category (skip the final category level)
        edges.push({
          id: `${parentPath}-${itemId}`,
          source: parentPath,
          target: itemId,
          type: 'smoothstep',
          animated: false,
        });
      });
    });

    // Calculate positions for hierarchical layout with proper parent-child alignment
    const maxLevel = Math.max(...Object.keys(levelNodes).map(Number));
    const startY = 80;
    
    // Position nodes level by level, ensuring children are positioned under their parents
    for (let levelIndex = 0; levelIndex <= maxLevel; levelIndex++) {
      const nodesAtLevel = levelNodes[levelIndex] || [];
      
      if (levelIndex === 0) {
        // Top level - position evenly across the screen
        const totalWidth = nodesAtLevel.length * (nodeWidth + nodeSpacing) - nodeSpacing;
        const startX = Math.max(100, (1200 - totalWidth) / 2); // Ensure minimum margin
        
        nodesAtLevel.forEach((node, index) => {
          node.position = {
            x: startX + index * (nodeWidth + nodeSpacing),
            y: startY + levelIndex * levelHeight
          };
        });
      } else if (levelIndex === maxLevel) {
        // Memory items - position under their direct parents
        const itemsByParent = {};
        nodesAtLevel.forEach(node => {
          const parentPath = node.data.parentPath;
          if (!itemsByParent[parentPath]) {
            itemsByParent[parentPath] = [];
          }
          itemsByParent[parentPath].push(node);
        });
        
        Object.entries(itemsByParent).forEach(([parentPath, items]) => {
          const parentNode = nodes.find(n => n.id === parentPath);
          if (parentNode) {
            const parentX = parentNode.position.x;
            const itemsWidth = items.length * (nodeWidth + nodeSpacing) - nodeSpacing;
            const startX = parentX - itemsWidth / 2 + nodeWidth / 2;
            
            items.forEach((item, index) => {
              item.position = {
                x: startX + index * (nodeWidth + nodeSpacing),
                y: startY + levelIndex * levelHeight
              };
            });
          }
        });
              } else {
          // Category nodes - position under their parents but avoid overlaps
          const nodesByParent = {};
          
          nodesAtLevel.forEach(node => {
            // Find parent by looking at the node's path
            const nodePathSegments = node.id.split('/');
            const parentPath = nodePathSegments.slice(0, -1).join('/');
            const parentKey = parentPath || 'root';
            
            if (!nodesByParent[parentKey]) {
              nodesByParent[parentKey] = [];
            }
            nodesByParent[parentKey].push(node);
          });
          
          // Track all positioned nodes at this level to avoid overlaps
          const positionedNodes = [];
          
          Object.entries(nodesByParent).forEach(([parentKey, childNodes]) => {
            if (parentKey === 'root') {
              // These are top-level children, spread them out
              const totalWidth = childNodes.length * (nodeWidth + nodeSpacing) - nodeSpacing;
              const startX = Math.max(100, (1200 - totalWidth) / 2);
              
              childNodes.forEach((node, index) => {
                const position = {
                  x: startX + index * (nodeWidth + nodeSpacing),
                  y: startY + levelIndex * levelHeight
                };
                node.position = position;
                positionedNodes.push({ node, x: position.x });
              });
            } else {
              // Position under their parent
              const parentNode = nodes.find(n => n.id === parentKey);
              if (parentNode) {
                const parentX = parentNode.position.x;
                const childrenWidth = childNodes.length * (nodeWidth + nodeSpacing) - nodeSpacing;
                let startX = parentX - childrenWidth / 2 + nodeWidth / 2;
                
                // Check for overlaps with already positioned nodes
                childNodes.forEach((node, index) => {
                  let proposedX = startX + index * (nodeWidth + nodeSpacing);
                  
                  // Check if this position overlaps with any existing nodes
                  let hasOverlap = true;
                  let attempts = 0;
                  const maxAttempts = 10;
                  
                  while (hasOverlap && attempts < maxAttempts) {
                    hasOverlap = positionedNodes.some(positioned => 
                      Math.abs(positioned.x - proposedX) < nodeWidth + 20 // 20px minimum gap
                    );
                    
                    if (hasOverlap) {
                      // Find the rightmost positioned node and place after it
                      const rightmostX = Math.max(...positionedNodes.map(p => p.x));
                      proposedX = rightmostX + nodeWidth + nodeSpacing;
                    }
                    attempts++;
                  }
                  
                  const position = {
                    x: proposedX,
                    y: startY + levelIndex * levelHeight
                  };
                  node.position = position;
                  positionedNodes.push({ node, x: position.x });
                });
              }
            }
          });
          
          // Sort positioned nodes by x position for clean layout
          positionedNodes.sort((a, b) => a.x - b.x);
        }
    }

    return { nodes, edges };
  };

  const updateVisibleNodes = (allNodes, allEdges, expandedCategories) => {
    // Filter nodes: show all category nodes and only memory items whose parents are expanded
    const visibleNodes = allNodes.filter(node => {
      if (node.data.type === 'category') {
        // Update the label based on expansion state
        if (node.data.isExpandable) {
          const isExpanded = expandedCategories.has(node.id);
          node.data.label = `${node.data.originalLabel} (${node.data.childCount}) ${isExpanded ? '−' : '+'}`;
          // Add data attributes for CSS styling
          node['data-expandable'] = 'true';
          node['data-expanded'] = isExpanded ? 'true' : 'false';
        } else {
          node['data-expandable'] = 'false';
          node['data-expanded'] = 'false';
        }
        return true; // Always show category nodes
      } else if (node.data.type === 'memory-item') {
        // Only show memory items if their parent is expanded
        return expandedCategories.has(node.data.parentPath);
      }
      return true;
    });

    // Filter edges: only show edges between visible nodes
    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
    const visibleEdges = allEdges.filter(edge => 
      visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
    );

    setNodes(visibleNodes);
    setEdges(visibleEdges);
  };

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  const onNodeClick = useCallback((event, node) => {
    if (node.data.type === 'category' && node.data.isExpandable) {
      // Toggle expansion for category nodes, but don't set selected node
      setExpandedCategories(prev => {
        const newExpanded = new Set(prev);
        if (newExpanded.has(node.id)) {
          newExpanded.delete(node.id);
        } else {
          newExpanded.add(node.id);
        }
        // Update visible nodes with new expansion state
        updateVisibleNodes(allNodes, allEdges, newExpanded);
        return newExpanded;
      });
    } else if (node.data.type === 'memory-item') {
      // Only set selected node for memory items (leaf nodes)
      setSelectedNode(node);
    }
  }, [allNodes, allEdges]);

  const renderNodeDetails = () => {
    // Only render details for memory items (sidebar only shows for memory items now)
    if (!selectedNode || selectedNode.data.type !== 'memory-item') {
      return null;
    }

    const { data } = selectedNode;
    const memoryItem = data.memoryItem;
    const itemDetails = memoryItem ? getItemDetails(memoryItem) : null;
    
    return (
      <div className="node-details-panel">
        <h3>{data.label}</h3>
        
        {memoryItem && (
          <div className="memory-item-details">
            {memoryType === 'episodic' && memoryItem.timestamp && (
              <div className="detail-section">
                <h4>Timestamp</h4>
                <p>{new Date(memoryItem.timestamp).toLocaleString()}</p>
              </div>
            )}
            
            {memoryType === 'procedural' && memoryItem.steps && (
              <div className="detail-section">
                <h4>Steps</h4>
                <ol>
                  {memoryItem.steps.map((step, index) => (
                    <li key={index}>{step}</li>
                  ))}
                </ol>
              </div>
            )}
            
            {memoryType === 'semantic' && memoryItem.tags && memoryItem.tags.length > 0 && (
              <div className="detail-section">
                <h4>Tags</h4>
                <p>{memoryItem.tags.join(', ')}</p>
              </div>
            )}
            
            {memoryType === 'semantic' && memoryItem.created_at && (
              <div className="detail-section">
                <h4>Created</h4>
                <p>{new Date(memoryItem.created_at).toLocaleString()}</p>
              </div>
            )}
            
            {itemDetails.summary && (
              <div className="detail-section">
                <h4>Summary</h4>
                <p>{itemDetails.summary}</p>
              </div>
            )}
            
            {itemDetails.details && (
              <div className="detail-section">
                <h4>Details</h4>
                <p>{itemDetails.details}</p>
              </div>
            )}

          <button 
            className="sidebar-close-button"
            onClick={() => setSelectedNode(null)}
            title="Close sidebar"
          >
            {t('uploadExport.form.close')}
          </button>

          </div>
        )}
        

      </div>
    );
  };

  if (loading) {
    return <div className="memory-tree-loading">{t('memory.states.loadingTree', { type: memoryType })}</div>;
  }

  if (error) {
    return <div className="memory-tree-error">{t('memory.states.treeError', { error })}</div>;
  }

  return (
    <div className="memory-tree-visualization">
      <div className="graph-container">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          fitView
          minZoom={0.1}
          maxZoom={2}
        >
          <Controls />
          <MiniMap 
            nodeColor={(node) => {
              switch (node.data?.type) {
                case 'category': return '#4ecdc4';
                case 'memory-item': return '#45b7d1';
                default: return '#96ceb4';
              }
            }}
          />
          <Background variant="dots" gap={12} size={1} />

        </ReactFlow>
      </div>
      {selectedNode && selectedNode.data.type === 'memory-item' && (
        <div className="details-sidebar">
          {renderNodeDetails()}
        </div>
      )}
    </div>
  );
};

export default MemoryTreeVisualization; 