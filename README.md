# 🚀 Passo a Passo para Reprodução dos Resultados do HW1 🚀

---

## 💻 1. Criar o `.venv` e instalar as dependências

Primeiro, configure o seu ambiente virtual para isolar o projeto.

Crie o ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente:

* No **macOS / Linux**:

```bash
source venv/bin/activate
```

* No **Windows (PowerShell / CMD)**:

```powershell
.\venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

> Se o Jupyter Notebook não estiver no `requirements.txt`, instale-o também:
>
> ```bash
> pip install notebook
> ```

---

## 📊 2. Rodar os códigos para analisar as métricas estatísticas

Execute os scripts de análise monovariada para gerar as estatísticas descritivas e os histogramas/boxplots iniciais.

* **Análise Incondicional (geral):**

```bash
python HW1/class_unconditional.py
```

* **Análise Condicional (por classe):**

```bash
python HW1/class_conditional.py
```

---

## 🔗 3. Rodar o código para analisar a correlação

Execute este script para gerar a matriz de correlação e os gráficos de dispersão.

```bash
python HW1/bivariate_analysis.py
```

---

## ✨ 4. Rodar o código para normalizar os dados

Execute este script para aplicar as transformações (Yeo-Johnson + Z-score) e salvar os dados processados. Esses dados transformados serão usados nos próximos passos.

```bash
python HW1/save_data_transformation.py
```

---

## 🌀 5. Aplicar as funções do PCA em um ambiente Jupyter Notebook

Para uma análise mais interativa dos Componentes Principais (PCA):

1. Inicie o Jupyter Notebook:

```bash
jupyter notebook
```

2. Crie um novo notebook (`.ipynb`).

3. Dentro do notebook:

   * Importe as funções necessárias do diretório `HW1/utils/` (ex.: `pca_calculation`, `pca_biplot`, `pca_scree_plot`).
   * Carregue os dados normalizados, por exemplo:

     ```
     HW1/data_transformations/data_yeojohnson_zscore.csv
     ```
   * Use as funções importadas para calcular o PCA, plotar o *scree plot* e o *biplot* interativamente.

---

## 🎯 6. Rodar o código para realizar a detecção de outliers

Finalmente, execute o script de detecção de outliers, que usará os dados normalizados e os scores do PCA para identificar e plotar os outliers (Figura 10).

```bash
python HW1/outlier.py
```

---


