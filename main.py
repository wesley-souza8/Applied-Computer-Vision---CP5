import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
import json
import contextlib
import io
import os

# ============================================================
# CONFIGURAÇÕES
# ============================================================
VIDEOS = [
    r"videos\v_JugglingBalls_g05_c01.avi",
    r"videos\levantandopeso01.mp4",
    r"videos\v_Lunges_g03_c03.avi"
]
NUMERO_QUADROS = 32
TAMANHO_IMAGEM_SLOWFAST = 224
ALPHA = 4
ARQUIVO_CLASSES = "kinetics_classnames.json"

# ============================================================
# INICIALIZAÇÃO DE MODELOS
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

print("Carregando YOLOv8...")
yolo_model = YOLO('yolov8n.pt')

print("Carregando Modelo de Segmentação (DeepLabV3 ResNet50)...")
seg_model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT).to(device)
seg_model.eval()

seg_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Carregando SlowFast...")
with contextlib.redirect_stderr(io.StringIO()):
    slowfast_model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True).to(device)
slowfast_model.eval()

with open(ARQUIVO_CLASSES, "r", encoding="utf-8") as f:
    classes_kinetics = json.load(f)
id_para_classe = {int(v): k for k, v in classes_kinetics.items()}

# ============================================================
# FUNÇÕES SLOWFAST
# ============================================================
def normalizar_video(video):
    media = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    desvio = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    return (video - media) / desvio

def criar_caminhos(video, alpha):
    caminho_fast = video
    total_quadros = video.shape[1]
    indices_slow = torch.linspace(0, total_quadros - 1, total_quadros // alpha).long()
    caminho_slow = torch.index_select(video, dim=1, index=indices_slow)
    return [caminho_slow, caminho_fast]

def processar_slowfast(buffer_quadros):
    video = np.stack(buffer_quadros)
    video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0
    video = normalizar_video(video)
    caminhos = criar_caminhos(video, ALPHA)
    caminhos = [c.unsqueeze(0).to(device) for c in caminhos]
    
    with torch.no_grad():
        logits = slowfast_model(caminhos)
        probabilidades = torch.softmax(logits, dim=1)
        prob, indice = torch.max(probabilidades, dim=1)
        
    return id_para_classe.get(indice.item(), f"Classe {indice.item()}"), prob.item()

# ============================================================
# PROCESSAMENTO DOS VÍDEOS SIMULTANEAMENTE
# ============================================================
def main():
    caps = []
    outs = []
    janelas = []
    buffers_sf = []
    acoes = []
    confiancas = []
    
    for caminho_video in VIDEOS:
        if not os.path.exists(caminho_video):
            print(f"Vídeo não encontrado: {caminho_video}")
            continue

        nome_base = os.path.splitext(os.path.basename(caminho_video))[0]
        video_out = f"output_{nome_base}.avi"

        cap = cv2.VideoCapture(caminho_video)
        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0

        out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'XVID'), fps, (largura, altura))
        
        caps.append(cap)
        outs.append(out)
        janelas.append(f"CP5 - {nome_base}")
        buffers_sf.append([])
        acoes.append("Aguardando...")
        confiancas.append(0.0)

    print("\n--- Iniciando processamento SIMULTÂNEO dos vídeos ---")
    
    while True:
        videos_ativos = 0
        
        for i in range(len(caps)):
            if not caps[i].isOpened():
                continue
                
            ret, frame = caps[i].read()
            if not ret:
                caps[i].release()
                outs[i].release()
                try:
                    cv2.destroyWindow(janelas[i])
                except:
                    pass
                continue
                
            videos_ativos += 1
            frame_anotado = frame.copy()
            altura_frame, largura_frame = frame.shape[:2]
            
            # 1. YOLO
            resultados_yolo = yolo_model(frame, classes=[0], verbose=False)
            maior_area = 0
            melhor_bbox = None
            
            if len(resultados_yolo) > 0 and len(resultados_yolo[0].boxes) > 0:
                for box in resultados_yolo[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > maior_area:
                        maior_area = area
                        melhor_bbox = (x1, y1, x2, y2)
            
            if melhor_bbox:
                x1, y1, x2, y2 = melhor_bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(largura_frame, x2), min(altura_frame, y2)
                
                cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_anotado, "Pessoa (YOLO)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 2. Segmentação
                recorte = frame[y1:y2, x1:x2]
                if recorte.size > 0:
                    recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
                    input_tensor = seg_transform(recorte_rgb).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        saida_seg = seg_model(input_tensor)['out'][0]
                        mascara = saida_seg.argmax(0).byte().cpu().numpy()
                        
                    mascara_pessoa = (mascara == 15).astype(np.uint8)
                    cor_mascara = np.zeros_like(recorte)
                    cor_mascara[:, :, 0] = 255
                    mascara_colorida = cv2.bitwise_and(cor_mascara, cor_mascara, mask=mascara_pessoa)
                    alpha = 0.5
                    recorte_com_mascara = cv2.addWeighted(recorte, 1 - alpha, mascara_colorida, alpha, 0)
                    frame_anotado[y1:y2, x1:x2] = np.where(mascara_colorida > 0, recorte_com_mascara, frame_anotado[y1:y2, x1:x2])
                    
            # 3. SlowFast
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_redimensionado = cv2.resize(frame_rgb, (TAMANHO_IMAGEM_SLOWFAST, TAMANHO_IMAGEM_SLOWFAST))
            buffers_sf[i].append(frame_redimensionado)
            
            if len(buffers_sf[i]) == NUMERO_QUADROS:
                acoes[i], confiancas[i] = processar_slowfast(buffers_sf[i])
                buffers_sf[i] = buffers_sf[i][16:]
                
            texto_acao = f"Acao: {acoes[i]} ({confiancas[i]*100:.1f}%)"
            cv2.putText(frame_anotado, texto_acao, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Exibir a janela individualmente
            frame_display = cv2.resize(frame_anotado, (largura_frame * 2, altura_frame * 2))
            cv2.imshow(janelas[i], frame_display)
            
            outs[i].write(frame_anotado)
            
        if videos_ativos == 0:
            break
            
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("\n[!] Processamento interrompido pelo usuário.")
            break

    for cap in caps:
        if cap.isOpened(): cap.release()
    for out in outs:
        out.release()
        
    cv2.destroyAllWindows()
    print("\nTODOS OS VÍDEOS FORAM PROCESSADOS E SALVOS COM SUCESSO!")

if __name__ == "__main__":
    main()
