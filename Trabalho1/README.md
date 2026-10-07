# Classificação Multirrótulo e Combinação de Modelos na Predição de Reações Adversas a Medicamentos

Código, dados, resultados e figuras do Trabalho 1 de Aprendizado de Máquina 2 (UFSCar, 2026/2).
Autor: Vitor Rodrigues da Mata (RA 831591) — vitormata@estudante.ufscar.br

O artigo (template SBC/ENIAC 2026) está em [`artigo.pdf`](artigo.pdf).

Temas da disciplina: **classificação multirrótulo** (Aula 02) e **combinação de modelos** (Aula 01).
Base de dados: **SIDER** (versão MoleculeNet), 1.427 fármacos × 27 classes de reações adversas (MedDRA).

## Estrutura

```
artigo.pdf                artigo completo
data/sider.csv            base SIDER (SMILES + 27 rótulos binários)
data/features.npz         atributos gerados por src/features.py
src/features.py           padronização das moléculas e cálculo de descritores RDKit + chaves MACCS
src/mlmethods.py          implementação dos métodos: Prior, BR, LP, CC, ensembles (EBR/ECC), RAkEL,
                          BR empilhado (stacking) e ML-kNN
src/evaluation.py         métricas multirrótulo, t-teste corrigido (Nadeau-Bengio), Holm, diversidade (Q)
src/eda.py                estatísticas da base e figuras exploratórias
src/prelim_features.py    experimento preliminar de escolha da representação molecular
src/run_main.py           experimento principal (10 dobras, RF e Regressão Logística)
src/run_rakel_k.py        sensibilidade do RAkEL ao tamanho k (k = 1 ... 27)
src/run_lp_decoding.py    LP com decodificação pela moda vs. marginal
src/analyze.py            agrega resultados, testes estatísticos, tabelas LaTeX, macros
                          numéricas do artigo (results/numbers.tex) e figuras
src/slide_figs.py         versões das figuras para os slides
src/run_all.sh            executa todos os experimentos em sequência
results/                  resultados por dobra (.pkl), resumos (.csv/.json) e tabelas (.tex)
figures/                  figuras do artigo (.pdf/.png); figures/slides/ = versões para a apresentação
```

## Como reproduzir

```bash
pip install -r requirements.txt
cd src
python features.py          # ~1 min
python eda.py
python prelim_features.py   # ~8 min
./run_all.sh                # ~3 h em 2 núcleos (retomável: dobras prontas são puladas)
```

A base foi obtida de
`https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/originals/sider.csv`
(cópia do arquivo `sider.csv` distribuído pelo MoleculeNet; SHA-256
`71efc6ac4ca82d6545bc512509863281be0f17234afd7adaf71a71899988302e`).

Sementes fixas: divisão em dobras `random_state=42`; Random Forest `random_state=0`.

## Principais resultados (médias em 10 dobras)

Ver `results/summary_main.csv`, `results/significance_vs_BR.csv`, `results/diversity.csv`,
`results/rakel_k_summary.csv` e `results/lp_decoding_summary.csv`. As probabilidades de teste
de cada método e dobra estão em `results/main/*.pkl`, o que permite refazer a análise sem
rodar os experimentos de novo.
