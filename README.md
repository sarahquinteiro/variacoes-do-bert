# Variações do BERT — BERT vs DistilBERT

Experimento e página de demonstração do seminário **Variações do BERT: RoBERTa, ALBERT,
DistilBERT** — Equipe 5, Processamento de Linguagem Natural, Fatec Osasco (DSM).

Gustavo de Sousa Lima · Karine Fernandes e Silva · Maxwell Alves · Sarah Quinteiro

**Página ao vivo:** https://sarahquinteiro.github.io/variacoes-do-bert/

A pergunta do experimento é a que o artigo do DistilBERT levanta: **quanto custa, em qualidade,
cortar metade das camadas de um BERT?** Treinamos os dois modelos na mesma tarefa de
classificação de sentimento em português e medimos qualidade, latência, tempo de treino e
tamanho.

---

## Resultado

Média e desvio-padrão sobre três sementes. Tarefa binária balanceada — o acaso dá 0,500.

| | BERT | DistilBERT |
|---|---|---|
| Acurácia | 0,869 ± 0,011 | **0,891 ± 0,010** |
| F1-macro | 0,869 ± 0,012 | **0,891 ± 0,010** |
| Tempo de treino | 40,9 s | **23,5 s** |
| Latência (CPU, 1 frase) | 170,7 ms | **86,1 ms** |
| Tamanho em disco | 679 MB | **516 MB** |
| Parâmetros | 178 M | **135 M** |

**Leitura:** 102,5% da acurácia retida · latência −50% · treino −43% · disco −24%.

A diferença de acurácia entre os dois é **menor que a variação entre as nossas próprias
sementes**. Com este conjunto de teste não é possível afirmar que um é mais preciso que o
outro: o resultado é empate em qualidade, com ganho real e consistente em custo. O DistilBERT
ter ficado nominalmente acima do BERT é a mesma barra de erro dita de outro jeito — quando a
diferença cabe dentro do ruído, ela cai para qualquer um dos lados.

### Por que só dois modelos, se o tema tem três variações

Porque `bert-base-multilingual-cased` → `distilbert-base-multilingual-cased` é o **único par em
que a arquitetura é a única variável**: um foi destilado do outro, com o mesmo corpus e o mesmo
vocabulário. Qualquer diferença de qualidade só pode vir da destilação.

As alternativas confundem a medida. O RoBERTa multilíngue (XLM-R, ~300 M) tem quase o dobro dos
parâmetros do mBERT (~178 M), então um ganho dele seria indistinguível de um ganho de tamanho.
E ALBERT multilíngue praticamente não existe fora do mALBERT ([arXiv:2403.18338](https://arxiv.org/abs/2403.18338)), um lançamento de pesquisa treinado só em Wikipédia.

Separamos, então: **qualidade no par limpo, custo nas quatro**. As quatro variações foram
cronometradas na mesma máquina, nos modelos monolíngues em inglês — a base de comparação dos
artigos originais e a única em que as quatro existem lado a lado:

| | Parâmetros | Camadas | Disco | Latência CPU |
|---|---|---|---|---|
| BERT | 109,5 M | 12 | 418 MB | 179,9 ms |
| RoBERTa | 124,6 M | 12 | 476 MB | 182,8 ms |
| ALBERT | 11,7 M | 12 | 45 MB | 175,7 ms |
| DistilBERT | 67,0 M | 6 | 256 MB | **89,9 ms** |

O ALBERT tem **um décimo** dos parâmetros do BERT e praticamente a mesma latência. O DistilBERT
tem seis vezes mais parâmetros que o ALBERT e é o dobro mais rápido. É o resultado que resume o
seminário: **parâmetro é memória, camada é tempo.**

BERT e RoBERTa têm arquitetura idêntica, e por isso servem de aferição da medida — os 1,6% que
separam os dois confirmam que a cronometragem está limpa.

---

## O conjunto de dados, e o erro que a gente cometeu primeiro

Mil frases em português brasileiro, balanceadas, em duas camadas: **316 escritas à mão** pela
equipe (avaliações de produto, restaurante, aplicativo e serviço, com negação, ironia e
ressalva) e 684 compostas por combinação de sujeitos e predicados dentro de cada família de
assunto.

A segunda camada existe por um motivo que vale registrar. A primeira versão do experimento
usava só as 316 escritas à mão, e o treino **empacava em 0,65 de acurácia** — quase o acaso.
Mexer em época e taxa de aprendizado não resolveu.

O problema eram os dados. Escrever 316 frases todas diferentes faz cada marcador de sentimento
aparecer duas ou três vezes no conjunto inteiro: com 237 frases de treino, o modelo via `ótimo`
quatro vezes e `péssimo` duas. Sinal fraco demais para amarrar a palavra ao rótulo em poucas
épocas. A confirmação veio de um classificador clássico de TF-IDF nas mesmas frases, que também
ficou em 0,52 — se um método que só conta palavras não extrai nada dali, o gargalo não é a
arquitetura.

Com a expansão, as palavras que aparecem uma única vez caíram de 55% para 42% do vocabulário e
o treino convergiu. As frases escritas à mão continuam no conjunto, e são as difíceis: o mesmo
TF-IDF acerta 0,88 no conjunto completo e apenas 0,62 nelas.

---

## Reproduzir

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarahquinteiro/variacoes-do-bert/blob/main/experimento_bert_vs_distilbert.ipynb)

1. Abra o `experimento_bert_vs_distilbert.ipynb` no Google Colab.
2. **Ambiente de execução → Alterar o tipo de ambiente** → acelerador **T4 GPU**. Sem GPU o
   notebook passa de uma hora.
3. **Executar tudo.** Leva de 10 a 15 minutos, dos quais ~2,5 GB são download de pesos.
4. A última célula baixa o `resultados_equipe5.json` e o gráfico.

O notebook confere o próprio resultado em dois pontos: avisa em letras garrafais se a acurácia
ficar abaixo de 0,70 (treino que não convergiu) e compara as latências de BERT e RoBERTa, que
têm arquitetura idêntica e precisam empatar — se divergirem mais de 15%, a medição pegou ruído
da CPU compartilhada e deve ser refeita.

O passo a passo detalhado, com o que esperar em cada etapa, está em
[`COMO-GERAR-AS-MEDICOES.md`](COMO-GERAR-AS-MEDICOES.md).

### Detalhes do treino

Laço de treino em PyTorch puro, sem a classe `Trainer` — a API dela muda entre versões, e o laço
explícito deixa visível o que está sendo medido.

- 4 épocas, lote 16, comprimento máximo 64 tokens (~184 passos de otimização)
- AdamW, taxa 3e-5, aquecimento de 10% dos passos seguido de decaimento linear
- Recorte de gradiente em 1,0
- Divisão treino/teste 75/25, estratificada, refeita a cada semente
- Sementes 42, 7 e 2024

A latência é medida **uma vez só, com os modelos intercalados** na mesma bateria — um, o outro,
um, o outro — depois de um aquecimento do processo. Medir um modelo de cada vez fazia o primeiro
pagar sozinho a inicialização do PyTorch: numa versão anterior o BERT saiu em 312 ms e o RoBERTa
em 167 ms, sendo que os dois têm exatamente a mesma arquitetura.

---

## A página

Arquivo único, estático, sem dependências além das fontes do Google — funciona offline com
fontes do sistema. Mostra o que os slides não mostram:

- **Confronto por frase** — a mesma frase pelos dois modelos, com rótulo, confiança e tempo de
  cada decisão. Abre numa frase em que os dois discordam.
- **Onde empatam e onde diferem** — os quatro gráficos, com desvio-padrão.
- **Leitura** — acurácia retida e queda de custo, calculadas dos valores em tela.
- **Qual variação cabe no seu projeto** — orçamento de latência e teto de memória filtrando as
  quatro variações. As faixas dos controles saem da própria medição carregada.
- **Cole o `resultados_equipe5.json`** — troca os valores de demonstração pela medição real.

A página abre com valores de demonstração, claramente etiquetados como tais. Colar o JSON do
notebook troca a etiqueta para *medição da equipe* — e o rótulo é derivado do campo `origem` do
arquivo, então ele não tem como mentir.

Para hospedar: [`HOSPEDAR-NO-GITHUB-PAGES.md`](HOSPEDAR-NO-GITHUB-PAGES.md).

---

## Arquivos

| Arquivo | O que é |
|---|---|
| `index.html` | a página da demonstração — é este arquivo que o GitHub Pages publica |
| `experimento_bert_vs_distilbert.ipynb` | o experimento completo, para rodar no Colab |
| `resultados_equipe5.json` | a medição que o notebook gerou |
| `COMO-GERAR-AS-MEDICOES.md` | passo a passo do Colab |
| `HOSPEDAR-NO-GITHUB-PAGES.md` | passo a passo da hospedagem |
| `roteiro-equipe5-variacoes-do-bert.md` | falas da apresentação |
| `slides-equipe5-variacoes-do-bert.md` | textos dos slides |

---

## O que este experimento não prova

- **Mil frases é pouco.** Por isso rodamos três sementes e reportamos desvio-padrão em vez de um
  número único. Com esse conjunto não dá para resolver uma diferença de dois pontos de acurácia.
- **As frases são nossas**, e parte delas é composta por combinação — o conjunto é mais regular
  que texto de usuário real. Serve para demonstrar o método, não para publicar um resultado.
- **A latência foi medida numa VM compartilhada do Colab.** O valor absoluto oscila entre
  execuções; o que é estável, e o que estamos reportando, é a razão entre os modelos, medida na
  mesma máquina e na mesma sessão.
- **Três sementes é o mínimo** para ter desvio-padrão. Cinco ou dez seriam melhores.
- A economia de disco ficou em 24%, abaixo dos 40% do artigo, porque o par que usamos é
  multilíngue: o vocabulário de 119.547 tokens gera uma matriz de *embeddings* que a destilação
  não corta e que pesa igual nos dois modelos. O artigo mediu o par monolíngue em inglês, com
  30.000 tokens.

---

## Referências

- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
- Lan et al. (2020). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.* ICLR 2020. [arXiv:1909.11942](https://arxiv.org/abs/1909.11942)
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- Hinton et al. (2015). *Distilling the Knowledge in a Neural Network.* [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

Trabalho acadêmico, sem fins comerciais.
