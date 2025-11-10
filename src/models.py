import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedStructuredClinicalTransformer(nn.Module):
    """IMPROVED Structured Clinical Transformer with Advanced Architecture"""
    def __init__(self, num_symptoms, num_animals, num_breeds, num_ages, num_clusters,
                 d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.config = {
            'num_symptoms': num_symptoms,
            'num_animals': num_animals,
            'num_breeds': num_breeds,
            'num_ages': num_ages,
            'num_clusters': num_clusters,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dropout': dropout
        }

        # Enhanced embedding layers
        self.symptom_embedding = nn.Embedding(num_symptoms + 1, d_model, padding_idx=0)
        self.animal_embedding = nn.Embedding(num_animals, d_model)
        self.breed_embedding = nn.Embedding(num_breeds, d_model)
        self.age_embedding = nn.Embedding(num_ages, d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)

        # Multi-scale feature extraction
        self.symptom_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.feature_norm = nn.LayerNorm(d_model)

        # Enhanced transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Advanced Clinical Prior Components
        self.clinical_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )

        self.cluster_embedding = nn.Embedding(num_clusters + 1, d_model, padding_idx=0)  
        self.severity_proj = nn.Linear(1, d_model)  

        # Multi-head attention pooling  
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  
        self.pool_norm = nn.LayerNorm(d_model)  

        # Enhanced fusion network  
        self.fusion_net = nn.Sequential(  
            nn.Linear(d_model * 2 + 128, d_model * 2),  
            nn.LayerNorm(d_model * 2),  
            nn.GELU(),  
            nn.Dropout(dropout),  
            nn.Linear(d_model * 2, d_model),  
            nn.LayerNorm(d_model),  
            nn.GELU(),  
            nn.Dropout(dropout),  
            nn.Linear(d_model, d_model // 2),  
            nn.GELU(),  
            nn.Dropout(dropout),  
            nn.Linear(d_model // 2, 1),  
            nn.Sigmoid()  
        )

        # Advanced feature processing  
        self.feature_processor = nn.Sequential(  
            nn.Linear(4, 64),  
            nn.GELU(),  
            nn.Dropout(dropout),  
            nn.Linear(64, 128),  
            nn.GELU(),  
            nn.Dropout(dropout)  
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

        # Initialize positional encoding
        nn.init.normal_(self.pos_encoding, mean=0, std=0.01)

    def forward(self, symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                animal_indices, breed_indices, age_indices, weight_values, symptom_counts, risk_counts):
        batch_size, seq_len = symptom_indices.shape

        # Enhanced symptom embeddings with multi-modal fusion
        symptom_embeds = self.symptom_embedding(symptom_indices)

        # Add clinical prior information
        clinical_embeds = self.clinical_proj(clinical_priors.unsqueeze(-1))
        cluster_embeds = self.cluster_embedding(symptom_clusters)
        severity_embeds = self.severity_proj(symptom_severities.unsqueeze(-1))

        # Multi-scale feature combination with residual connections
        symptom_embeds = (symptom_embeds + 0.4 * clinical_embeds + 
                         0.3 * cluster_embeds + 0.3 * severity_embeds)
        symptom_embeds = self.feature_norm(symptom_embeds)

        # Apply 1D convolution for local pattern extraction
        symptom_embeds = symptom_embeds.transpose(1, 2)
        symptom_embeds = self.symptom_conv(symptom_embeds)
        symptom_embeds = symptom_embeds.transpose(1, 2)
        symptom_embeds = self.feature_norm(symptom_embeds)

        # Positional encoding
        symptom_embeds = symptom_embeds + self.pos_encoding[:, :seq_len, :]

        # Enhanced animal context with attention
        animal_context = (
            self.animal_embedding(animal_indices) +
            self.breed_embedding(breed_indices) +
            self.age_embedding(age_indices)
        )

        # Combine features with animal context using attention
        animal_context_expanded = animal_context.unsqueeze(1).expand(-1, seq_len, -1)
        combined_embeds = symptom_embeds + animal_context_expanded

        # Transformer encoding with enhanced masking
        src_key_padding_mask = (symptom_indices == 0)
        encoded = self.transformer(combined_embeds, src_key_padding_mask=src_key_padding_mask)

        # Advanced multi-head attention pooling
        query = animal_context.unsqueeze(1)  # Use animal context as query

        attended, attention_weights = self.attention_pool(
            query, encoded, encoded,
            key_padding_mask=src_key_padding_mask
        )
        attention_pooled = attended.squeeze(1)
        attention_pooled = self.pool_norm(attention_pooled)

        # Traditional pooling as fallback
        mask = src_key_padding_mask.unsqueeze(-1)
        encoded_masked = encoded.masked_fill(mask, 0.0)
        mean_pooled = encoded_masked.sum(dim=1) / (~mask).sum(dim=1).clamp(min=1.0)
        max_pooled, _ = encoded_masked.max(dim=1)

        # Dynamic pooling combination
        pooled = 0.6 * attention_pooled + 0.25 * mean_pooled + 0.15 * max_pooled

        # Enhanced additional features processing
        additional_features = torch.stack([
            symptom_counts / 10.0,
            risk_counts[:, 0] / 5.0,
            risk_counts[:, 1] / 5.0,
            torch.tanh(weight_values)
        ], dim=1)

        processed_features = self.feature_processor(additional_features)

        # Final prediction with enhanced feature combination
        combined = torch.cat([pooled, animal_context, processed_features], dim=1)
        prediction = self.fusion_net(combined)

        return prediction

class VeterinaryLSTM(nn.Module):
    """LSTM Model for Veterinary Health Assessment"""
    def __init__(self, num_symptoms, num_animals, num_breeds, num_ages,
                 hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layers
        self.symptom_embedding = nn.Embedding(num_symptoms + 1, 64, padding_idx=0)
        self.animal_embedding = nn.Embedding(num_animals, 32)
        self.breed_embedding = nn.Embedding(num_breeds, 32)
        self.age_embedding = nn.Embedding(num_ages, 16)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32 + 32 + 16 + 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, symptom_indices, symptom_severities, symptom_clusters, clinical_priors,
                animal_indices, breed_indices, age_indices, weight_values, symptom_counts, risk_counts):
        batch_size, seq_len = symptom_indices.shape

        # Symptom embeddings
        symptom_embeds = self.symptom_embedding(symptom_indices)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(symptom_embeds)

        # Use last hidden state from both directions
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        lstm_combined = torch.cat([hidden_forward, hidden_backward], dim=1)

        # Animal context
        animal_context = torch.cat([
            self.animal_embedding(animal_indices),
            self.breed_embedding(breed_indices),
            self.age_embedding(age_indices)
        ], dim=1)

        # Additional features
        additional_features = torch.stack([
            symptom_counts / 10.0,
            risk_counts[:, 0] / 5.0,
            risk_counts[:, 1] / 5.0,
            torch.tanh(weight_values)
        ], dim=1)

        # Final prediction
        combined = torch.cat([lstm_combined, animal_context, additional_features], dim=1)
        prediction = self.fusion_net(combined)

        return prediction