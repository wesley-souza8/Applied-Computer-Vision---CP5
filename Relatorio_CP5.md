# Relatório de Visão Computacional - CheckPoint 5
**Tema:** Detecção de Ações de Pessoas com Yolo, U-Net e SlowFast

## 1. Experimentos Realizados
Para a realização dos experimentos, foram processados dois vídeos distintos utilizando a pipeline de YOLOv8 (para a caixa), DeepLabV3 - U-Net Proxy (para a máscara) e SlowFast (para classificação de ação em janelas consecutivas de 32 quadros). 
Os vídeos escolhidos foram: `v_JugglingBalls_g05_c01.avi` e `levantandopeso01.mp4`.

### Tabela Comparativa: Três Momentos de Cada Vídeo

| Vídeo | Momento | Ação Visual Real | Predição do SlowFast | Confiança | Observação do Comportamento |
|-------|---------|------------------|----------------------|-----------|-----------------------------|
| **JugglingBalls** | 1. Início | Pessoa parada, bolas na mão | `juggling balls` | Alta | O modelo capta a intenção inicial pela postura e contexto. |
| **JugglingBalls** | 2. Meio | Malabarismo ativo e rápido | `juggling balls` | Média/Alta | Leve oscilação de confiança devido ao desfoque (blur) das mãos. |
| **JugglingBalls** | 3. Fim | Apanhando as bolas, parando | `juggling balls` | Baixa (~49%) | **[Queda]** A quebra do padrão cíclico temporal reduz a confiança drasticamente. |
| **Levantamento** | 1. Início | Puxada da barra do chão | `deadlifting` | ~87% | Precisão alta, postura característica bem delineada. |
| **Levantamento** | 2. Meio | Erguendo a barra p/ o alto | `clean and jerk` / `snatch` | ~58% | **[Mudança de Classe]** O modelo altera a predição ao reconhecer outra fase anatômica do esporte. |
| **Levantamento** | 3. Fim | Repouso, segurando a barra | `deadlifting` | ~31% | **[Queda]** O fim da dinâmica de movimento derruba a confiança. |

---

## 2. Análise e Respostas aos Questionamentos

**A caixa acompanha a pessoa?**
**Sim.** O modelo YOLOv8 manteve a caixa delimitadora (*bounding box*) de forma contínua e altamente estável em ambos os vídeos. Mesmo durante o agachamento profundo da atleta (levantamento de peso) ou a movimentação rápida e cruzada dos braços (malabarismo), a caixa enquadrou ativamente o indivíduo sem perder o rastreamento em nenhum momento.

**A máscara acompanha seu contorno?**
**Sim, com grande precisão.** A segmentação semântica, projetada e recortada a partir da região detectada pelo YOLO, conseguiu isolar e colorir os pixels corporais fielmente na maior parte do tempo. Houve variações mínimas nas bordas apenas nos instantes de muito *blur* (borrão de movimento) gerado pelas mãos do malabarista, mas o contorno primário (tronco, pernas, cabeça) acompanhou perfeitamente a silhueta da pessoa.

**A ação prevista corresponde ao que aparece no trecho?**
**Sim.** Em *JugglingBalls*, a predição foi exatamente correspondente ao malabarismo de forma ininterrupta. No *Levantamento de peso*, o modelo não apenas acertou a atividade geral, como demonstrou sensibilidade às fases do exercício: identificando corretamente os momentos de `deadlifting` (peso morto/tração inicial) e alterando dinamicamente para `clean and jerk` / `snatch` (arranque) quando a barra foi erguida acima dos ombros.

**A confiança cai ou a classe muda quando a ação começa ou termina?**
**Sim, e este foi o fenômeno mais notável documentado (explicando os casos de "erro" ou instabilidade):**
1. **Mudança de Classe:** No meio do vídeo de Levantamento de Peso, a classe mudou de `deadlifting` para `clean and jerk`. O modelo detectou a transição do movimento, gerando uma fragmentação natural da predição, o que reduziu um pouco a confiança (caindo para 58%), pois as poses se sobrepõem muito.
2. **Queda de Confiança no Término:** Ao final de ambos os vídeos, a confiança despenca vertiginosamente. No levantamento de peso, quando a atleta termina o exercício e fica imóvel segurando a barra, a confiança desaba de incríveis 87% para apenas **31.6%**. No malabarismo ocorre o mesmo: quando o indivíduo para de jogar as bolas, a confiança escorrega para a casa dos 40~50%. Como o modelo SlowFast realiza convoluções 3D dependentes de fluxo temporal, a ausência da dinâmica motora (ou seja, quando a ação termina e entra em repouso estático) destrói a "certeza" do classificador, comprovando perfeitamente a teoria estrutural da rede.
