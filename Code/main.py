"""
Experimento: Classificação de Filmes IMDb - Overfitting vs Generalização
=========================================================================

Este script demonstra:
1. Como preparar dados reais (IMDb Top 250 Movies)
2. Treinar modelos com diferentes profundidades
3. Comparar overfitting vs generalização
4. Analisar resultados com matriz de confusão

Autor: Professor Alexandre Montanha
Dataset: IMDb Top 250 Movies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class ExperimentoIMDb:
    """
    Classe para experimento de classificação de filmes IMDb
    """
    
    def __init__(self, caminho_csv='Database/base_imdb.csv'):
        """
        Inicializa o experimento
        
        Args:
            caminho_csv: Caminho para o arquivo CSV do IMDb
        """
        self.caminho_csv = caminho_csv
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.modelos = {}
        
    def carregar_dados(self):
        """
        Carrega o dataset IMDb e exibe informações básicas
        """
        print("=" * 70)
        print("📽️  CARREGANDO DATASET IMDB TOP 250 MOVIES")
        print("=" * 70)
        
        self.df = pd.read_csv(self.caminho_csv)
        
        print(f"\n✅ Dataset carregado com sucesso!")
        print(f"📊 Total de filmes: {len(self.df)}")
        print(f"📋 Colunas disponíveis: {list(self.df.columns)}")
        print("\n🎬 Primeiros filmes:")
        print(self.df[['Title', 'Year', 'Rating', 'Duration', 'Genres']].head(10))
        
        return self
    
    def preparar_dados(self):
        """
        Prepara os dados para treinamento
        """
        print("\n" + "=" * 70)
        print("🔧 PREPARANDO DADOS PARA MACHINE LEARNING")
        print("=" * 70)
        
        # Criar cópia para trabalhar
        df_prep = self.df.copy()
        
        # 1. Criar variável alvo: Rating alto (≥ 8.5) vs médio (< 8.5)
        df_prep['Categoria'] = df_prep['Rating'].apply(
            lambda x: 'Alta' if x >= 8.5 else 'Média'
        )
        
        print("\n📊 Distribuição das categorias:")
        print(df_prep['Categoria'].value_counts())
        print(f"   - Porcentagem 'Alta': {(df_prep['Categoria'] == 'Alta').mean() * 100:.1f}%")
        print(f"   - Porcentagem 'Média': {(df_prep['Categoria'] == 'Média').mean() * 100:.1f}%")
        
        # 2. Extrair ano
        df_prep['Year'] = pd.to_numeric(df_prep['Year'], errors='coerce')
        
        # 3. Converter duração (ex: "2h 22m" -> 142 minutos)
        def converter_duracao(duracao_str):
            try:
                minutos = 0
                if 'h' in str(duracao_str):
                    partes = str(duracao_str).replace('m', '').split('h')
                    minutos = int(partes[0]) * 60
                    if len(partes) > 1 and partes[1].strip():
                        minutos += int(partes[1].strip())
                else:
                    minutos = int(str(duracao_str).replace('m', '').strip())
                return minutos
            except:
                return np.nan
        
        df_prep['Duration_Minutes'] = df_prep['Duration'].apply(converter_duracao)
        
        # 4. Extrair primeiro gênero
        df_prep['Primary_Genre'] = df_prep['Genres'].apply(
            lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown'
        )
        
        # 5. Codificar Certificate (PG, PG-13, R, etc.)
        df_prep['Certificate'] = df_prep['Certificate'].fillna('Not Rated')
        
        # Remover linhas com valores ausentes
        df_prep = df_prep.dropna(subset=['Year', 'Duration_Minutes', 'Certificate', 'Primary_Genre'])
        
        print(f"\n✅ Dados após limpeza: {len(df_prep)} filmes")
        
        # 6. Criar features numéricas e categóricas
        # Codificar variáveis categóricas
        le_cert = LabelEncoder()
        le_genre = LabelEncoder()
        
        df_prep['Certificate_Encoded'] = le_cert.fit_transform(df_prep['Certificate'])
        df_prep['Genre_Encoded'] = le_genre.fit_transform(df_prep['Primary_Genre'])
        
        # Criar década
        df_prep['Decade'] = (df_prep['Year'] // 10) * 10
        
        print("\n📋 Features criadas:")
        print("   1. Year (Ano de lançamento)")
        print("   2. Duration_Minutes (Duração em minutos)")
        print("   3. Certificate_Encoded (Classificação etária)")
        print("   4. Genre_Encoded (Gênero principal)")
        print("   5. Decade (Década de lançamento)")
        
        # 7. Selecionar features e target
        features = ['Year', 'Duration_Minutes', 'Certificate_Encoded', 
                   'Genre_Encoded', 'Decade']
        X = df_prep[features]
        y = df_prep['Categoria']
        
        # 8. Dividir em treino e teste (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✅ Divisão treino/teste:")
        print(f"   📚 Treino: {len(self.X_train)} filmes ({len(self.X_train)/len(df_prep)*100:.1f}%)")
        print(f"   🧪 Teste:  {len(self.X_test)} filmes ({len(self.X_test)/len(df_prep)*100:.1f}%)")
        
        # Guardar encoders para referência
        self.le_cert = le_cert
        self.le_genre = le_genre
        self.df_prep = df_prep
        
        return self
    
    def treinar_modelos(self):
        """
        Treina dois modelos: um com overfitting e outro otimizado
        """
        print("\n" + "=" * 70)
        print("🤖 TREINANDO MODELOS DE ÁRVORE DE DECISÃO")
        print("=" * 70)
        
        # Modelo A: Profundidade ilimitada (overfitting)
        print("\n📊 Modelo A - Árvore Profunda (max_depth=None)")
        print("   Expectativa: Alto overfitting")
        modelo_a = DecisionTreeClassifier(
            max_depth=None,
            random_state=42,
            min_samples_split=2
        )
        modelo_a.fit(self.X_train, self.y_train)
        
        # Modelo B: Profundidade limitada (generalização)
        print("\n📊 Modelo B - Árvore Rasa (max_depth=3)")
        print("   Expectativa: Melhor generalização")
        modelo_b = DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )
        modelo_b.fit(self.X_train, self.y_train)
        
        self.modelos = {
            'Modelo A (Overfitting)': modelo_a,
            'Modelo B (Generalização)': modelo_b
        }
        
        print("\n✅ Modelos treinados com sucesso!")
        
        return self
    
    def avaliar_modelos(self):
        """
        Avalia e compara os modelos
        """
        print("\n" + "=" * 70)
        print("📈 AVALIAÇÃO DOS MODELOS")
        print("=" * 70)
        
        resultados = []
        
        for nome, modelo in self.modelos.items():
            print(f"\n{'─' * 70}")
            print(f"🔍 {nome}")
            print(f"{'─' * 70}")
            
            # Predições
            y_train_pred = modelo.predict(self.X_train)
            y_test_pred = modelo.predict(self.X_test)
            
            # Acurácias
            acc_train = accuracy_score(self.y_train, y_train_pred)
            acc_test = accuracy_score(self.y_test, y_test_pred)
            
            print(f"\n📊 Acurácia no TREINO: {acc_train * 100:.2f}%")
            print(f"📊 Acurácia no TESTE:  {acc_test * 100:.2f}%")
            print(f"⚠️  Diferença (Overfitting): {(acc_train - acc_test) * 100:.2f} pontos percentuais")
            
            # Interpretação
            diferenca = acc_train - acc_test
            if diferenca > 0.15:
                print(f"🔴 OVERFITTING DETECTADO! Modelo memorizou o treino.")
            elif diferenca > 0.05:
                print(f"🟡 Leve overfitting. Aceitável em muitos casos.")
            else:
                print(f"🟢 Boa generalização! Modelo balanceado.")
            
            # Relatório de classificação
            print(f"\n📋 Relatório de Classificação (Teste):")
            print(classification_report(self.y_test, y_test_pred))
            
            resultados.append({
                'Modelo': nome,
                'Acurácia Treino': acc_train,
                'Acurácia Teste': acc_test,
                'Diferença': diferenca,
                'Predições Teste': y_test_pred
            })
        
        self.resultados = resultados
        return self
    
    def visualizar_comparacao(self):
        """
        Cria visualizações comparativas dos modelos
        """
        print("\n" + "=" * 70)
        print("📊 GERANDO VISUALIZAÇÕES COMPARATIVAS")
        print("=" * 70)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Comparação de Acurácias
        ax1 = plt.subplot(2, 3, 1)
        modelos_nomes = [r['Modelo'] for r in self.resultados]
        acc_treino = [r['Acurácia Treino'] * 100 for r in self.resultados]
        acc_teste = [r['Acurácia Teste'] * 100 for r in self.resultados]
        
        x = np.arange(len(modelos_nomes))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, acc_treino, width, label='Treino', color='skyblue')
        bars2 = ax1.bar(x + width/2, acc_teste, width, label='Teste', color='coral')
        
        ax1.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Comparação: Acurácia Treino vs Teste', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Modelo A\n(Overfitting)', 'Modelo B\n(Generalização)'])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 105])
        
        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Diferença (Overfitting)
        ax2 = plt.subplot(2, 3, 2)
        diferencas = [r['Diferença'] * 100 for r in self.resultados]
        cores = ['red' if d > 15 else 'orange' if d > 5 else 'green' for d in diferencas]
        
        bars = ax2.bar(range(len(modelos_nomes)), diferencas, color=cores, alpha=0.7)
        ax2.set_ylabel('Diferença (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Overfitting: Treino - Teste', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(modelos_nomes)))
        ax2.set_xticklabels(['Modelo A', 'Modelo B'])
        ax2.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Alto Overfitting')
        ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Overfitting Leve')
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        # Adicionar valores
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3 e 4. Matrizes de Confusão
        for idx, resultado in enumerate(self.resultados):
            ax = plt.subplot(2, 3, 3 + idx)
            
            cm = confusion_matrix(self.y_test, resultado['Predições Teste'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Alta', 'Média'],
                       yticklabels=['Alta', 'Média'],
                       ax=ax, cbar_kws={'label': 'Quantidade'})
            
            ax.set_ylabel('Valor Real', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predição', fontsize=11, fontweight='bold')
            ax.set_title(f'Matriz de Confusão\n{resultado["Modelo"]}', 
                        fontsize=12, fontweight='bold')
        
        # 5. Visualização das árvores
        ax5 = plt.subplot(2, 3, 5)
        plot_tree(self.modelos['Modelo A (Overfitting)'], 
                 filled=True, 
                 feature_names=['Year', 'Duration', 'Certificate', 'Genre', 'Decade'],
                 class_names=['Alta', 'Média'],
                 max_depth=3,
                 ax=ax5,
                 fontsize=7)
        ax5.set_title('Modelo A - Árvore (3 primeiros níveis)', fontsize=11, fontweight='bold')
        
        ax6 = plt.subplot(2, 3, 6)
        plot_tree(self.modelos['Modelo B (Generalização)'], 
                 filled=True,
                 feature_names=['Year', 'Duration', 'Certificate', 'Genre', 'Decade'],
                 class_names=['Alta', 'Média'],
                 ax=ax6,
                 fontsize=8)
        ax6.set_title('Modelo B - Árvore Completa', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('experimento_imdb_resultados.png', dpi=300, bbox_inches='tight')
        print("✅ Gráfico salvo como 'experimento_imdb_resultados.png'")
        plt.show()
        
        return self
    
    def conclusao(self):
        """
        Apresenta a conclusão do experimento
        """
        print("\n" + "=" * 70)
        print("🎓 CONCLUSÃO DO EXPERIMENTO")
        print("=" * 70)
        
        modelo_a = self.resultados[0]
        modelo_b = self.resultados[1]
        
        print("\n📌 Modelo A (max_depth=None):")
        print(f"   Treino: {modelo_a['Acurácia Treino']*100:.2f}%")
        print(f"   Teste:  {modelo_a['Acurácia Teste']*100:.2f}%")
        print(f"   ❌ Problema: OVERFITTING ({modelo_a['Diferença']*100:.2f}% de diferença)")
        print(f"   → O modelo memorizou os filmes de treino, mas não generalizou.")
        
        print("\n📌 Modelo B (max_depth=3):")
        print(f"   Treino: {modelo_b['Acurácia Treino']*100:.2f}%")
        print(f"   Teste:  {modelo_b['Acurácia Teste']*100:.2f}%")
        print(f"   ✅ Solução: BOA GENERALIZAÇÃO ({modelo_b['Diferença']*100:.2f}% de diferença)")
        print(f"   → O modelo aprendeu padrões reais que funcionam em novos filmes.")
        
        print("\n" + "=" * 70)
        print("🏆 RESPOSTA DA QUESTÃO DO QUIZ")
        print("=" * 70)
        print("\n✅ Escolha o MODELO B!")
        print("\n💡 Por quê?")
        print("   1. Menor diferença entre treino e teste (menos overfitting)")
        print("   2. Melhor performance em dados nunca vistos")
        print("   3. Regras mais simples e interpretáveis")
        print("   4. Maior confiabilidade em produção")
        
        print("\n⚠️  Lição aprendida:")
        print("   → Acurácia de 100% no treino é um ALERTA, não uma vitória!")
        print("   → Sempre avalie o modelo em dados de TESTE")
        print("   → Prefira modelos que GENERALIZAM sobre os que MEMORIZAM")
        
        print("\n" + "=" * 70)
        
        return self
    
    def executar_experimento_completo(self):
        """
        Executa o pipeline completo do experimento
        """
        print("\n🎬 INICIANDO EXPERIMENTO COMPLETO: CLASSIFICAÇÃO IMDB")
        print("=" * 70)
        
        self.carregar_dados()
        self.preparar_dados()
        self.treinar_modelos()
        self.avaliar_modelos()
        self.visualizar_comparacao()
        self.conclusao()
        
        print("\n✅ EXPERIMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
        
        return self


def main():
    """
    Função principal
    """
    # Criar e executar experimento
    experimento = ExperimentoIMDb()
    experimento.executar_experimento_completo()
    
    # Exemplos adicionais
    print("\n" + "=" * 70)
    print("🔬 EXEMPLOS ADICIONAIS")
    print("=" * 70)
    
    print("\n📝 Para experimentar outros parâmetros:")
    print("""
    # Exemplo 1: Árvore muito profunda (overfitting extremo)
    modelo_profundo = DecisionTreeClassifier(max_depth=20)
    
    # Exemplo 2: Árvore muito rasa (underfitting)
    modelo_raso = DecisionTreeClassifier(max_depth=1)
    
    # Exemplo 3: Balanceamento ideal
    modelo_otimizado = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )
    """)
    
    print("\n📚 Próximos passos:")
    print("   1. Experimente com outros algoritmos (SVM, Random Forest)")
    print("   2. Adicione mais features (década, número de gêneros)")
    print("   3. Use validação cruzada para resultados mais robustos")
    print("   4. Analise quais filmes o modelo erra mais")


if __name__ == "__main__":
    main()
