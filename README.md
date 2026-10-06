# Applied Computer Vision - CheckPoint 5
**Detecção de Ações de Pessoas com Yolo-UNet-SlowFast**

Projeto acadêmico para avaliação de Visão Computacional. O sistema lê vídeos, detecta o enquadramento de pessoas (*Bounding Boxes*), cria uma máscara exata de segmentação e avalia a ação contínua através do contexto temporal.

## Arquitetura
1. **YOLO (YOLOv8):** Responsável pela detecção da classe "pessoa" e recorte inicial da região.
2. **DeepLabV3 (U-Net Proxy):** Responsável por segmentar o corpo isolando os pixels relevantes.
3. **SlowFast:** Captura *sliding windows* de 32 quadros e analisa convolucionalmente as ações temporais e sua probabilidade.

## Como Executar
1. Instale as dependências executando: `pip install ultralytics pytorchvideo opencv-python torch torchvision`
2. Garanta que o arquivo `kinetics_classnames.json` e a pasta de vídeos de testes (`videos/`) estão presentes na raiz do repositório.
3. Rode `python main.py`

O script abrirá uma janela mostrando em tempo real o processamento e as marcações (*boxes*, máscara e texto de predição), exportando o resultado final para um arquivo `output_*.avi`.

## Relatório
Acesse o arquivo `Relatorio_CP5.md` para ver a comparação formal exigida pelo roteiro do professor em relação aos acertos e quebras de estado da IA.
