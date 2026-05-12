import torch
import torch.nn as nn

class ClinicalEvolver(nn.Module):
    """
    [Bio-COT 5.0 Core Module] 整合到 3.2
    Dynamic Clinical Query Evolution (CoT)
    """
    def __init__(self, visual_dim=768, clinical_dim=256, dropout=0.2):
        super().__init__()
        
        self.visual_attention = nn.Sequential(
            nn.Linear(visual_dim, 1),
            nn.Tanh()
        )
        
        self.feedback_proj = nn.Sequential(
            nn.Linear(visual_dim, clinical_dim),
            nn.Dropout(dropout) 
        )
        
        self.gru = nn.GRUCell(input_size=clinical_dim, hidden_size=clinical_dim)
        self.norm = nn.LayerNorm(clinical_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_map, prev_clinical_state):
        """
        visual_map: [B, N, D_v]
        prev_clinical_state: [B, D_c]
        Returns:
            new_clinical_state: [B, D_c]
        """
        # 1. Attention Pooling
        attn_scores = self.visual_attention(visual_map) 
        attn_weights = torch.softmax(attn_scores, dim=1)
        visual_feedback = (visual_map * attn_weights).sum(dim=1)
        
        # 2. Project
        feedback_input = self.feedback_proj(visual_feedback)
        
        # 3. Update State
        new_clinical_state = self.gru(feedback_input, prev_clinical_state)
        
        # 4. Norm & Dropout
        return self.dropout(self.norm(new_clinical_state))

