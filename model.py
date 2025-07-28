import torch 
import torch.nn as nn 
import torch.nn.functional as F 

"""
Re-implementation of BERT-base: from BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
arxiv.org/abs/1810.04805
L = 12, H=768, A=12. 

This implementation is reverse engineered from the HuggingFace bert-base-uncased model:
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 x BertLayer(
        (attention): BertAttention(
          (self): BertSdpaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
"""



class BertEmbeddings(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim, 
        max_len=512, 
        type_vocab_size=2, 
        dropout=0.1
    ):
        super().__init__()
        # padding idx set to 0, [PAD]
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(num_embeddings=max_len, embedding_dim=embed_dim)
        self.token_type_embeddings = nn.Embedding(num_embeddings=type_vocab_size, embedding_dim=embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids, token_type_ids=None): 
        # input ids is (batch, shape) tensor.
        B, L = input_ids.shape
        # obtain word embedding mapping 
        word_embeds = self.word_embeddings(input_ids)

        # create the positions tensor
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        pos_embeds = self.position_embeddings(positions)
    
        # token type embeddings 
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        type_embeds = self.token_type_embeddings(token_type_ids) 

        # element wise addition of positional, and token type embeddings 
        x = word_embeds + pos_embeds + type_embeds
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x
    

class BertSelfOutput(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # residual connection, as per AIAYN paper, Vaswani et al., 2017
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states 

class BertSdpaSelfAttention(nn.Module): 
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads 
        self.dropout_p = dropout
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # qkv matrices 
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x): 
        # we rearrange the input tensor to prepare it for multi head self attention.
        # (batch, seq_len, embed_dim) to (batch, num_heads, seq_len, head_dim)
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)

    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        attn_scores = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attention_mask, 
                dropout_p=self.dropout_p if self.training else 0.0
        )

        attn_scores = attn_scores.permute(0, 2, 1, 3).contiguous()
        attn_scores = attn_scores.view(hidden_states.size(0), hidden_states.size(1), self.embed_dim)
        return attn_scores
        
class BertAttention(nn.Module): 
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.self = BertSdpaSelfAttention(embed_dim, num_heads, dropout)
        self.output = BertSelfOutput(embed_dim, dropout)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states) # add and norm
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, embed_dim=768, intermediate_dim=3072):
        super().__init__()
        self.dense = nn.Linear(embed_dim, intermediate_dim)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states): 
        hidden_states = self.dense(hidden_states) 
        return self.intermediate_act_fn(hidden_states)
    
            
class BertOutput(nn.Module): 
    def __init__(self, embed_dim=768, intermediate_dim=3072, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states) 
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(input_tensor + hidden_states) # residual 
        return hidden_states 

class BertLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, intermediate_dim=3072, dropout=0.1):
        super().__init__()
        self.attention = BertAttention(embed_dim, num_heads, dropout)
        self.intermediate = BertIntermediate(embed_dim, intermediate_dim) 
        self.output = BertOutput(embed_dim, intermediate_dim, dropout)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        


class BertEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, intermediate_dim=3072, dropout=0.1, num_layers=12):
        super().__init__() 
        self.layer = nn.ModuleList(
            [BertLayer(embed_dim, num_heads, intermediate_dim, dropout) for i in range(num_layers)]
        )
    def forward(self, x, attention_mask=None):
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x


        
class BertPooler(nn.Module): 
    def __init__(self, embed_dim):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        # get the first token in every batch, [CLS] token. 
        cls_token = hidden_states[:, 0]
        pooled_output = self.dense(cls_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output


 
class BertModel(nn.Module): 
    # base model definition that outputs logits. 
    # a task head must be added for training (eg MLM or classifier head)
    def __init__(
        self, 
        embed_dim=768, 
        intermediate_dim=3072, 
        vocab_size=30000,
        max_len=512, 
        type_vocab_size=2, 
        dropout=0.1, 
        num_heads=12, 
        num_layers=12, 
    ):
        super().__init__()
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            max_len=max_len, 
            type_vocab_size=type_vocab_size, 
            dropout=dropout
        )
        self.encoder = BertEncoder(embed_dim, num_heads, intermediate_dim, dropout, num_layers)
        self.pooler = BertPooler(embed_dim)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is not None:
            # Convert to shape (batch, 1, 1, seq_len)
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoder_output)
        
        return {
            'last_hidden_state': encoder_output, 
            'pooled_output': pooled_output 
        }


class BertMLMHead(nn.Module):
    # maps logits to a distribution length = vocab size 
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        # bias is redundant to softmax 
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        

    def forward(self, hidden_states): 
        x = self.dense(hidden_states) 
        x = self.act_fn(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x 

        
class BertClassifierHead(nn.Module): 
    # classifier head. 
    def __init__(self, embed_dim, dropout=0.1, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(embed_dim // 2, num_classes)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim // 2)

        
    def forward(self, x):
        x = self.dropout(self.fc1(x))
        x = self.layer_norm(self.act_fn(x))
        x = self.fc2(x)
        return x
    



# finally: two wrapper classes for different training tasks 
class BertForMaskedLM(nn.Module):
    def __init__(self, embed_dim=768, vocab_size=30000, **kwargs):
        super().__init__()
        self.bert = BertModel(embed_dim=embed_dim, vocab_size=vocab_size, **kwargs)
        self.mlm_head = BertMLMHead(embed_dim, vocab_size)
        # weight tying 
        self.mlm_head.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask) 
        prediction_scores = self.mlm_head(outputs['last_hidden_state'])
        loss = None 

        # for the pretraining task, if we have some labels, we can return the loss right away. 
        if labels is not None: 
            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            loss = criterion(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))
        return {
            'loss': loss, 
            'logits': prediction_scores
        }

class BertForSequenceClassification(nn.Module): 
    def __init__(self, num_classes, embed_dim=768, vocab_size=30000, **kwargs):
        super().__init__()
        self.bert = BertModel(embed_dim=embed_dim, vocab_size=vocab_size, **kwargs)
        self.clf = BertClassifierHead(embed_dim=embed_dim, num_classes=num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        # just get the pooled output from the BertModel head [CLS] token
        logits = self.clf(outputs['pooled_output'])

        loss = None 

        # same as before 
        if labels is not None: 
            criterion = nn.CrossEntropyLoss() 
            loss = criterion(logits, labels)
        return {
            'loss': loss, 
            'logits': logits
        }

# test a forward pass 
if __name__ == "__main__":
    model = BertModel()
    batch_size, seq_len = 2,10
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))

    with torch.no_grad(): 
        output = model(input_ids, attention_mask=attention_mask)
        print(f"Last hidden state shape: {output['last_hidden_state'].shape}")
        print(f"Pooler output shape: {output['pooled_output'].shape}")
        
    # test mlm head 
    mlm_model = BertForMaskedLM()
    mlm_out = mlm_model(input_ids, attention_mask)
    print(f"MLM logits: {mlm_out['logits'].shape}")

    # test clf head 
    clf_model = BertForSequenceClassification(num_classes=3)
    clf_out = clf_model(input_ids, attention_mask)
    print(f"Classifier logits: {clf_out['logits'].shape}")
    
    
        
        