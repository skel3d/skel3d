import torch
import torch.nn as nn
import math

#from torch_geometric.nn import GATConv  # Example using GAT from torch_geometric

################## IMAGE-BASED REPRESENTATION ##################

class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim

        # Create the positional encodings for height and width
        pe = torch.zeros(embed_dim, height, width)
        
        y_pos = torch.arange(0, height, dtype=torch.float).unsqueeze(1).repeat(1, width)  # Shape (height, width)
        x_pos = torch.arange(0, width, dtype=torch.float).unsqueeze(0).repeat(height, 1)  # Shape (height, width)
        
        # Div term for sine and cosine
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        
        # Apply sin to even indices and cos to odd indices
        pe[0::2, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(1))  # Apply to even indices
        pe[1::2, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(1))  # Apply to odd indices
        
        # Register pe as a buffer so it can be used in the forward pass
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, embed_dim, height, width)
        Add positional encoding to the input tensor.
        """
        return x + self.pe.unsqueeze(0)
    
class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProjection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class ImageSkeletonFusionTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6, height=32, width=32):
        super(ImageSkeletonFusionTransformer, self).__init__()
        self.pos_encoding = PositionalEncoding2D(embed_dim, height, width)
        
        
        # Transformer for image-skeleton fusion
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim*2, nhead=num_heads)
        self.image_skeleton_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, source_image_embedding, source_skeleton_embedding):
        # Apply positional encoding
        source_image_embedding = self.pos_encoding(source_image_embedding)
        source_skeleton_embedding = self.pos_encoding(source_skeleton_embedding)

        # Flatten spatial dimensions
        B, C, H, W = source_image_embedding.shape
        source_image_flat = source_image_embedding.view(B, C, H * W).permute(2, 0, 1)
        source_skeleton_flat = source_skeleton_embedding.view(B, C, H * W).permute(2, 0, 1)

        # Cross-attention between image and skeleton
        fused_image_skeleton = self.image_skeleton_transformer(torch.cat([source_skeleton_flat, source_image_flat], dim=-1))
        
        return fused_image_skeleton.permute(1, 2, 0).view(B, C*2, H, W)  # Reshape to 2D
    
class SkeletonSkeletonFusionTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6, height=32, width=32):
        super(SkeletonSkeletonFusionTransformer, self).__init__()
        self.pos_encoding = PositionalEncoding2D(embed_dim, height, width)
        
        
        # Transformer for skeleton-skeleton fusion
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim*2, nhead=num_heads)
        self.skeleton_skeleton_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, source_skeleton_embedding, target_skeleton_embedding):
        # Apply positional encoding
        source_skeleton_embedding = self.pos_encoding(source_skeleton_embedding)
        target_skeleton_embedding = self.pos_encoding(target_skeleton_embedding)

        # Flatten spatial dimensions
        B, C, H, W = source_skeleton_embedding.shape
        source_skeleton_flat = source_skeleton_embedding.view(B, C, H * W).permute(2, 0, 1)
        target_skeleton_flat = target_skeleton_embedding.view(B, C, H * W).permute(2, 0, 1)

        # Cross-attention between source and target skeleton
        fused_skeletons = self.skeleton_skeleton_transformer(torch.cat([target_skeleton_flat, source_skeleton_flat], dim=-1))
        
        return fused_skeletons.permute(1, 2, 0).view(B, C*2, H, W)  # Reshape to 2D
    

'''
# Fuse source image and source skeleton
    image_skeleton_fusion = self.image_skeleton_transformer(source_image, source_skeleton)
    
    # Fuse source skeleton and target skeleton
    skeleton_skeleton_fusion = self.skeleton_skeleton_transformer(source_skeleton, target_skeleton)
    
    # Combine the embeddings (concatenation or add
    combined_embedding = torch.cat([image_skeleton_fusion, skeleton_skeleton_fusion], dim=-1)
    or fused_embedding = image_skeleton_fusion + skeleton_skeleton_fusion
    
'''



############## GRAPH-BASED SKELETON REPRESENTATION ##############

class SkeletonGraphRepresentation(nn.Module):
    def __init__(self, embed_dim=256):
        super(SkeletonGraphRepresentation, self).__init__()
        # Linear layer to project 2D coordinates to embedding space
        self.joint_embedding = nn.Linear(2, embed_dim)  # 2 for (x, y) coordinates
        
        # Edge (bone) representation (can be adjacency matrix or edge features)
        # In this case, you could represent bones by adjacency information (i.e., connections between joints)
    
    def forward(self, joint_coords, bones):
        """
        joint_coords: (B, num_joints, 2) tensor representing (x, y) coordinates of joints
        bones: Adjacency matrix or list of edges representing bone connections
        """
        # Step 1: Embed the joint coordinates
        joint_embeddings = self.joint_embedding(joint_coords)  # (B, num_joints, embed_dim)
        
        # Step 2: Process the bones (edges) if needed for attention or graph modeling (GNN)
        # Example: Apply graph-based operations on the joints based on the bone connections
        
        return joint_embeddings  # Return the embedded joint representations
    
class SkeletonGraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim, heads=8):
        super(SkeletonGraphAttention, self).__init__()
        self.gat = GATConv(input_dim, output_dim // heads, heads=heads)  # GAT for joint-bone interaction

    def forward(self, joint_embeddings, bone_edges):
        """
        joint_embeddings: (B, num_joints, input_dim) tensor representing joint embeddings
        bone_edges: Edge indices or adjacency matrix representing bone connections
        """
        # Apply graph attention between joints
        joint_embeddings = self.gat(joint_embeddings, bone_edges)
        return joint_embeddings
    
class ImageSkeletonFusion(nn.Module):
    def __init__(self, image_feature_extractor, skeleton_graph_attention, embed_dim=256):
        super(ImageSkeletonFusion, self).__init__()
        self.image_feature_extractor = image_feature_extractor  # Image feature extractor (e.g., from autoencoder)
        self.skeleton_graph_attention = skeleton_graph_attention  # Graph attention for skeletons
        self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)  # To fuse image and skeleton embeddings
    
    def forward(self, source_image, joint_coords, bone_edges):
        # Extract image features
        source_image_features = self.image_feature_extractor(source_image)
        
        # Extract skeleton features using graph attention
        skeleton_embeddings = self.skeleton_graph_attention(joint_coords, bone_edges)
        
        # Flatten and fuse both embeddings (e.g., via concatenation or addition)
        combined_embedding = torch.cat([source_image_features, skeleton_embeddings], dim=-1)
        fused_embedding = self.fusion_layer(combined_embedding)
        
        return fused_embedding  # Output fused embeddings for U-Net conditioning