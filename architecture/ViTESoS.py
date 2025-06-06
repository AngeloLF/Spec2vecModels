import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import torch.nn as nn
from einops import rearrange
import math




class PatchEmbedding(nn.Module):
    """
    Convertit une image en une séquence d'embeddings de patches.
    """
    def __init__(self, in_channels: int = 1, patch_size: tuple[int, int] = (16, 16),
                 emb_size: int = 768, img_size: tuple[int, int] = (128, 1024)):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        # La couche de convolution divise l'image en patches et les projette dans l'espace d'embedding.
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # Flatten à partir de la 2ème dimension pour obtenir (B, Emb_Size, Num_Patches)
            nn.Flatten(2)
        )
        # Calcul du nombre de patches en hauteur et en largeur
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.projection(x) # -> (B, Emb_Size, Num_Patches)
        x = x.transpose(1, 2)  # -> (B, Num_Patches, Emb_Size)
        return x

        

class MultiHeadSelfAttention(nn.Module):
    """
    Implémentation de l'attention multi-tête auto-attention.
    """
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # Chaque tête gère une partie de la dimension d'embedding
        self.head_dim = emb_size // num_heads
        if self.head_dim * num_heads != emb_size:
            raise ValueError(f"emb_size ({emb_size}) doit être divisible par num_heads ({num_heads})")

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.fc_out = nn.Linear(emb_size, emb_size) # Couche de sortie après concaténation des têtes
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim) # Pour scaler les scores d'attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Calcul des Q, K, V pour toutes les têtes
        # -> (B, S, H, D_head)
        keys = self.keys(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = self.queries(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.values(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transposition pour que les têtes soient la première dimension pour le calcul
        # -> (B, H, S, D_head)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Calcul de l'énergie (scores d'attention)
        # (B, H, S, D_head) @ (B, H, D_head, S) -> (B, H, S, S)
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # Appliquer l'attention aux valeurs et concaténer les sorties des têtes
        # (B, H, S, S) @ (B, H, S, D_head) -> (B, H, S, D_head)
        out = torch.matmul(attention, values)
        # Réarranger pour regrouper les dimensions des têtes et des séquences
        # -> (B, S, H, D_head) -> (B, S, Emb_Size)
        out = rearrange(out, 'b h s d -> b s (h d)') # Utilisation de einops

        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    """
    Couche Feed-Forward (MLP) dans un bloc Transformer.
    """
    def __init__(self, emb_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(), # Fonction d'activation GELU
            nn.Dropout(dropout),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Un bloc Transformer composé d'attention multi-tête et d'un MLP.
    """
    def __init__(self, emb_size: int, num_heads: int, ff_hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size) # Normalisation de couche avant l'attention
        self.attn = MultiHeadSelfAttention(emb_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_size) # Normalisation de couche avant le MLP
        self.ff = FeedForward(emb_size, ff_hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Connexions résiduelles et normalisation
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class VisionTransformerEncoder(nn.Module):
    """
    Encodeur du Vision Transformer.
    """
    def __init__(self, img_size: tuple[int, int] = (128, 1024), in_channels: int = 1,
                 patch_size: tuple[int, int] = (16, 16), emb_size: int = 768,
                 num_heads: int = 8, ff_hidden_size: int = 3072, num_layers: int = 12,
                 dropout: float = 0.0):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        
        # Positionnal embedding pour donner un sens de l'ordre aux patches
        # Ajout d'un terme pour le CLS token si utilisé, mais pas nécessaire pour le débruitage
        self.positional_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.patch_embedding(x) # -> (B, Num_Patches, Emb_Size)

        # Ajout de l'embedding de position
        x = x + self.positional_embedding
        x = self.dropout(x)

        # Passage à travers les blocs Transformer
        for block in self.transformer_blocks:
            x = block(x)
        return x # Sortie de l'encodeur: (B, Num_Patches, Emb_Size)

class ViTDecoder(nn.Module):
    """
    Décodeur pour reconstruire l'image à partir des embeddings de l'encodeur.
    Utilise des couches de convolution transposée pour remonter en résolution.
    """
    def __init__(self, emb_size: int = 768, patch_size: tuple[int, int] = (16, 16),
                 img_size: tuple[int, int] = (128, 1024), out_channels: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]

        # Les couches de déconvolution doivent "défaire" le découpage en patches.
        # Elles remonteront la résolution de (num_patches_h, num_patches_w) à (img_size[0], img_size[1]).
        # Le facteur de déconvolution est la taille du patch (par exemple, 16x16 pour un patch_size de 16x16)
        # Ici, on peut utiliser plusieurs couches pour une reconstruction plus progressive.
        
        # Exemple de couches de déconvolution pour remonter de (8, 64) à (128, 1024) avec patch_size=(16,16)
        # Facteur de scaling total = 128/8 = 16 (H) et 1024/64 = 16 (W)
        self.deconv = nn.Sequential(
            # De (emb_size, 8, 64) après reshape
            nn.ConvTranspose2d(emb_size, 512, kernel_size=2, stride=2), # -> (512, 16, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), # -> (256, 32, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # -> (128, 64, 512)
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, kernel_size=2, stride=2), # -> (out_channels, 128, 1024)
            # Pas de fonction d'activation finale comme Sigmoid/Tanh si les valeurs de pixels sont brutes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Num_Patches, Emb_Size)
        batch_size, num_patches, emb_size = x.shape

        # Remettre les embeddings dans une forme "image" (B, Emb_Size, H_patches, W_patches)
        # Utilisation de einops pour faciliter le reshape
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.num_patches_h, w=self.num_patches_w)

        # Passer à travers les couches de déconvolution
        x = self.deconv(x) # -> (B, Out_Channels, Img_H, Img_W)
        return x

class DenoisingViT(nn.Module):
    """
    Modèle complet de Vision Transformer Encoder-Decoder pour le débruitage d'images.
    """
    def __init__(self, img_size: tuple[int, int] = (128, 1024), in_channels: int = 1,
                 out_channels: int = 1, patch_size: tuple[int, int] = (16, 16),
                 emb_size: int = 768, num_heads: int = 8, ff_hidden_size: int = 3072,
                 num_layers: int = 6, dropout: float = 0.1): # num_layers réduit pour l'exemple
        super().__init__()
        self.encoder = VisionTransformerEncoder(img_size, in_channels, patch_size, emb_size,
                                             num_heads, ff_hidden_size, num_layers, dropout)
        self.decoder = ViTDecoder(emb_size, patch_size, img_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_features = self.encoder(x) # Encoder: (B, C, H, W) -> (B, Num_Patches, Emb_Size)
        denoised_image = self.decoder(encoded_features) # Decoder: (B, Num_Patches, Emb_Size) -> (B, Out_C, H, W)
        return denoised_image












class VitESoSDataset(Dataset):

    def __init__(self, image_dir, spectro_dir):
        self.image_dir = image_dir
        self.spectro_dir = spectro_dir
        self.input_files = sorted(os.listdir(image_dir))
        self.output_files = sorted(os.listdir(spectro_dir))

        # Une vérification essentielle : assure-toi que le nombre de fichiers correspond
        if len(self.input_files) != len(self.output_files):
            raise ValueError("Le nombre de fichiers d'entrée et de sortie doit être le même.")

        # Optionnel mais recommandé : si tes fichiers d'entrée et de sortie ont des noms correspondants
        # (par exemple, 'image_001.npy' dans les deux dossiers), tu peux vérifier ça ici.
        for i in range(len(self.input_files)):
            if self.input_files[i].split("_")[-1] != self.output_files[i].split("_")[-1]:
                print(f"Attention: Les noms de fichiers ne correspondent pas pour l'index {i}: {self.input_files[i]} vs {self.output_files[i]}")
            else:
                print(f"Ok pour {self.input_files[i]} vs {self.output_files[i]} ")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_filename = self.input_files[idx]
        output_filename = self.output_files[idx]

        input_filepath = os.path.join(self.image_dir, input_filename)
        output_filepath = os.path.join(self.spectro_dir, output_filename)

        # Charge les fichiers .npy
        noisy_image_np = np.load(input_filepath)
        clean_image_np = np.load(output_filepath)

        # Convertit les tableaux NumPy en tenseurs PyTorch.
        # Tes images sont 128x1024, donc NumPy va probablement les charger comme (H, W).
        # PyTorch attend (C, H, W). Puisque c'est un seul canal (gris), on ajoute une dimension.
        # .float() convertit en float32, ce qui est nécessaire pour les calculs de réseaux neuronaux.
        # Cette conversion ne normalise pas les valeurs ! Elles restent dans leur plage d'origine.
        noisy_image_tensor = torch.from_numpy(noisy_image_np).float().unsqueeze(0) # Ajoute la dimension du canal (1, 128, 1024)
        clean_image_tensor = torch.from_numpy(clean_image_np).float().unsqueeze(0) # Ajoute la dimension du canal (1, 128, 1024)

        # Vérification simple de la taille au chargement si tu veux être sûr
        if noisy_image_tensor.shape != (1, 128, 1024):
            raise ValueError(f"Image d'entrée {input_filename} n'a pas la bonne forme: {noisy_image_tensor.shape}")
        if clean_image_tensor.shape != (1, 128, 1024):
            raise ValueError(f"Image de sortie {output_filename} n'a pas la bonne forme: {clean_image_tensor.shape}")

        return noisy_image_tensor, clean_image_tensor



# if __name__ == "__main__":

#     folder_input = "image"
#     folder_output = "spectro"

#     sys.path.append("./")
#     datafold = sys.argv[1]

#     dataset = VitESoSDataset(image_dir=f"./results/output_simu/{datafold}/{folder_input}", spectro_dir=f"./results/output_simu/{datafold}/{folder_output}")
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Pour inspecter un lot de données chargé :
# # for noisy_batch, clean_batch in dataloader:
# #     print(f"Forme du lot d'entrée: {noisy_batch.shape}")  # Attendu: [batch_size, 1, 128, 1024]
# #     print(f"Forme du lot de sortie: {clean_batch.shape}") # Attendu: [batch_size, 1, 128, 1024]
# #     print(f"Type de données: {noisy_batch.dtype}")       # Attendu: torch.float32
# #     print(f"Exemple de valeurs de pixels (input): {noisy_batch[0, 0, 0, :5]}")
# #     print(f"Min/Max valeurs (input): {noisy_batch.min().item()}/{noisy_batch.max().item()}")
# #     break




if __name__ == "__main__":

    

    sys.path.append("./")
    datafold = sys.argv[1]

    folder_input = "image"
    folder_output = "spectro"

    # --- 1. Paramètres Généraux ---
    IMG_SIZE = (128, 1024)
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    BATCH_SIZE = 4
    NUM_SAMPLES = 10 # Nombre de paires d'images simulées
    PATCH_SIZE = (16, 16) # Doit diviser IMG_SIZE (128/16=8, 1024/16=64)

    # Paramètres du ViT (ajustables)
    EMB_SIZE = 256 # Dimension de l'embedding, plus petit pour un exemple rapide
    NUM_HEADS = 4
    FF_HIDDEN_SIZE = EMB_SIZE * 2
    NUM_LAYERS = 2 # Réduit pour un exemple rapide

    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {DEVICE}")

    DATA_DIR = f"./results/output_simu/{datafold}"
    NOISY_DIR = os.path.join(DATA_DIR, folder_input)
    CLEAN_DIR = os.path.join(DATA_DIR, folder_output)

    # --- 3. Initialisation du Dataset et DataLoader ---
    dataset = VitESoSDataset(image_dir=f"./results/output_simu/{datafold}/{folder_input}", spectro_dir=f"./results/output_simu/{datafold}/{folder_output}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset créé avec {len(dataset)} échantillons. DataLoader prêt.")

    # --- 4. Initialisation du Modèle, de la Perte et de l'Optimiseur ---
    model = DenoisingViT(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        patch_size=PATCH_SIZE,
        emb_size=EMB_SIZE,
        num_heads=NUM_HEADS,
        ff_hidden_size=FF_HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    # Pour le débruitage avec valeurs de pixels brutes, MSE (Mean Squared Error) est un excellent choix.
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Modèle, fonction de perte et optimiseur initialisés.")
    print(f"Nombre de paramètres du modèle: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- 5. Boucle d'entraînement (simplifiée) ---
    print("\nDébut de l'entraînement...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Mettre le modèle en mode entraînement
        running_loss = 0.0
        
        # tqdm est une bibliothèque pour afficher une barre de progression
        from tqdm import tqdm
        for batch_idx, (noisy_images, clean_images) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            noisy_images = noisy_images.to(DEVICE)
            clean_images = clean_images.to(DEVICE)

            # Zéro gradiente
            optimizer.zero_grad()

            # Forward pass
            outputs = model(noisy_images)

            # Calcul de la perte
            loss = criterion(outputs, clean_images)

            # Backward pass et optimisation
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy_images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

    print("\nEntraînement terminé.")

    # --- 6. Sauvegarde du modèle (optionnel) ---
    # model_save_path = "denoising_vit_model.pth"
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Modèle sauvegardé sous: {model_save_path}")

    # --- 7. Test du modèle et Visualisation (Exemple simple) ---
    # print("\nTest du modèle et visualisation d'un exemple...")
    # model.eval() # Mettre le modèle en mode évaluation
    # with torch.no_grad(): # Désactiver le calcul des gradients
    #     # Récupérer un échantillon du dataset pour la visualisation
    #     sample_noisy_img, sample_clean_img = dataset[0] 
        
    #     # Ajouter une dimension de batch et déplacer vers le device
    #     sample_noisy_img_batch = sample_noisy_img.unsqueeze(0).to(DEVICE)
        
    #     # Faire la prédiction
    #     denoised_output = model(sample_noisy_img_batch)
        
    #     # Déplacer le résultat vers le CPU et convertir en numpy
    #     denoised_output_np = denoised_output.squeeze(0).squeeze(0).cpu().numpy() # (1, 128, 1024) -> (128, 1024)

    #     # Afficher les images
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #     axes[0].imshow(sample_noisy_img.squeeze(0).numpy(), cmap='gray')
    #     axes[0].set_title("Image Bruitée (Input)")
    #     axes[0].axis('off')

    #     axes[1].imshow(sample_clean_img.squeeze(0).numpy(), cmap='gray')
    #     axes[1].set_title("Image Nette (Ground Truth)")
    #     axes[1].axis('off')

    #     axes[2].imshow(denoised_output_np, cmap='gray')
    #     axes[2].set_title("Image Débruitée (Modèle)")
    #     axes[2].axis('off')

    #     plt.suptitle("Résultats du Débruitage ViT")
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste le layout pour le titre
    #     plt.show()

    # --- 8. Nettoyage des données simulées (optionnel) ---
    # shutil.rmtree(DATA_DIR)
    # print(f"Dossier de données simulées '{DATA_DIR}' supprimé.")