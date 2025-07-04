import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import io
import base64
import os
import traceback
import glob
from gradio_client import Client, file, handle_file
import shutil
import tempfile
import sys
import requests
import time

sys.path.append(os.path.dirname(__file__))
# from processamento import extrair_dados_da_imagem
# from recomendacao import recomendar_roupas
#Try to import your custom functions - with error handling
try:
    from processamento import extrair_dados_da_imagem

    PROCESSAMENTO_AVAILABLE = True
except ImportError:
    st.error("⚠️ Módulo 'processamento' não encontrado. Algumas funcionalidades estarão limitadas.")
    PROCESSAMENTO_AVAILABLE = False

try:
    from recomendacao import recomendar_roupas

    RECOMENDACAO_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Módulo 'recomendacao' não encontrado. Usando versão simplificada.")
    RECOMENDACAO_AVAILABLE = False

# Page configuration
# Page configuration - Coloque st.set_page_config como a primeira chamada do Streamlit
st.set_page_config(
    page_title="VesteAI - Análise de Coloração",  # Nome do App
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS para aplicar os estilos desejados (Rosa Claro, Bege Claro, Texto Marrom/Preto)
# Cores:
# Rosa Claro: #FFDFD3 (usado no seu exemplo de homepage) ou #FFE5E9 (mais suave)
# Bege Claro: #FFF5E1 (usado no seu exemplo de homepage) ou #FDF5E6
# Marrom Escuro (texto): #5D4037
# Marrom Médio (texto secundário): #795548
# Preto (texto títulos): #000000 ou #333333 (quase preto)

custom_css = """
<style>
    /* Cor de fundo principal do corpo da aplicação */
    .stApp {
        background-color: #FFF5E1; /* Bege Claro */
    }

    /* Estilização do cabeçalho principal (se você usar st.title ou markdown h1) */
    .main-header, h1 {
        color: #000000 !important; /* Preto para o título principal */
        text-align: center;
        /* Se quiser um fundo diferente para o header, adicione aqui */
        /* background-color: #FFDFD3; /* Rosa Claro */
        /* padding: 1rem; */
        /* border-radius: 8px; */
    }

    /* Estilização de subtítulos e texto principal */
    h2, h3, h4, h5, h6 {
        color: #5D4037; /* Marrom Escuro para subtítulos */
    }

    p, .stMarkdown, .stText, .stAlert, label {
        color: #5D4037; /* Marrom Escuro para texto geral e labels */
    }

    /* Estilização da Sidebar */
    .stSidebar > div:first-child {
        background-color: #FFDFD3; /* Rosa Claro para o fundo da sidebar */
        /* border-right: 2px solid #E6C8B3; /* Bege um pouco mais escuro para borda */
    }

    .stSidebar .stMarkdown p, .stSidebar .stText, .stSidebar label, .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #5D4037 !important; /* Marrom Escuro para texto na sidebar */
    }
    .stSidebar .stButton>button {
        background-color: #5D4037;
        color: #FFFFFF;
        border-radius: 20px;
        border: 1px solid #5D4037;
    }
    .stSidebar .stButton>button:hover {
        background-color: #4E342E;
        color: #FFF5E1;
    }


    /* Estilização de Botões */
    .stButton>button {
        background-color: #5D4037; /* Marrom Escuro */
        color: #FFFFFF; /* Texto branco */
        border-radius: 20px; /* Botões arredondados */
        border: 1px solid #5D4037;
        padding: 0.5em 1em;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4E342E; /* Marrom mais escuro no hover */
        color: #FFF5E1; /* Texto bege claro no hover */
    }
    .stButton>button:focus:not(:active) {
        color: #FFF5E1; /* Mantém a cor do texto no foco */
        border-color: #4E342E;
    }


    /* Estilização de cards ou containers que você possa criar */
    .metric-card { /* Este é do seu CSS original, pode adaptar */
        background-color: #FFFFFF; /* Branco para cards, para destacar do bege */
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #FFDFD3; /* Borda rosa claro */
    }
    .metric-card label, .metric-card div[data-testid="stText"] {
        color: #5D4037 !important; /* Garante que o texto dentro dos cards seja marrom */
    }


    /* Ajustes em file uploader, selectbox etc. */
    .stFileUploader label, .stSelectbox label, .stTextInput label {
        color: #5D4037 !important; /* Marrom para labels de widgets */
    }

    /* Cor de fundo para caixas de código (st.code) */
    pre {
        background-color: #FFFAF0; /* Um bege bem clarinho, quase branco */
        color: #5D4037;
        border: 1px solid #FFDFD3;
        border-radius: 5px;
    }

    /* Cores para st.success, st.warning, st.error, st.info */
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    div[data-testid="stAlert"] p { /* Texto dentro dos alertas */
        color: #333333 !important; /* Ou uma cor mais escura que combine */
    }
    /* Para st.info - pode usar um tom de bege/marrom mais suave */
    /* div[data-baseweb="alert"][role="alert"] > div:first-child {
        background-color: #FFF9F0 !important;
    } */


    /* Melhorar contraste em tabelas (Pandas DataFrames) */
    .stDataFrame table {
        color: #5D4037; /* Texto marrom nas tabelas */
    }
    .stDataFrame th {
        background-color: #FFDFD3; /* Fundo rosa claro para cabeçalhos de tabela */
        color: #5D4037; /* Texto marrom para cabeçalhos */
    }
    .stDataFrame td, .stDataFrame th {
        border: 1px solid #E6C8B3; /* Borda bege mais escura */
    }

    /* Estilo para o expander */
    .streamlit-expanderHeader {
        color: #5D4037 !important; /* Marrom para o título do expander */
        background-color: #FFF9F0; /* Bege bem claro para o fundo do cabeçalho do expander */
        border-radius: 5px;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def pil_to_opencv(pil_image):
    """Convert PIL image to OpenCV format"""
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR (OpenCV uses BGR)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def opencv_to_base64(cv_img):
    """Convert OpenCV image to base64 for display in Streamlit"""
    _, buffer = cv2.imencode('.png', cv_img)
    img_base64 = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_base64}"


def opencv_to_pil(cv_img):
    """Convert OpenCV image to PIL using io buffer"""
    # Convert BGR to RGB
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    # Encode to PNG in memory
    _, buffer = cv2.imencode('.png', cv_img_rgb)
    # Use io.BytesIO to create a bytes buffer
    img_buffer = io.BytesIO(buffer)
    # Convert to PIL Image
    pil_img = Image.open(img_buffer)
    return pil_img


def create_downloadable_image(cv_img, filename="analysis_result.png"):
    """Create a downloadable image using io buffer"""
    # Convert to PIL first
    pil_img = opencv_to_pil(cv_img)

    # Create buffer for download
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)

    return buffer.getvalue(), filename


def criar_painel_cores(medidas):
    """Create a color panel showing extracted colors"""
    painel = np.full((400, 600, 3), (211, 223, 255), dtype=np.uint8)
    y_pos = 50

    def desenhar_texto_com_acentos(img_cv2, texto, pos, cor=(0, 0, 0), tamanho=20):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        base_path = os.path.dirname(os.path.abspath(__file__))
        fonte_path = os.path.join(base_path, "..", "data", "fonts", "arial.ttf")
        try:
            fonte = ImageFont.truetype(fonte_path, tamanho)
        except OSError:
            fonte = ImageFont.load_default()

        x, y = pos

        # Apenas dois pontos de contorno leves
        draw.text((x - 1, y - 1), texto, font=fonte, fill=cor)
        draw.text((x + 1, y + 1), texto, font=fonte, fill=cor)

        # Texto principal
        draw.text((x, y), texto, font=fonte, fill=cor)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Skin tone
    if 'tom_de_pele' in medidas:
        cv2.putText(painel, "Tom de Pele:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cor_pele = tuple(map(int, medidas['tom_de_pele']))
        cv2.rectangle(painel, (200, y_pos - 20), (300, y_pos + 20), cor_pele, -1)
        cv2.putText(painel, f"BGR: {list(cor_pele)}", (320, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 80

    # Hair tone
    if 'tom_de_cabelo' in medidas and not medidas.get('pouco_cabelo', True):
        cv2.putText(painel, "Tom de Cabelo:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cor_cabelo = tuple(map(int, medidas['tom_de_cabelo']))
        cv2.rectangle(painel, (200, y_pos - 20), (300, y_pos + 20), cor_cabelo, -1)
        cv2.putText(painel, f"BGR: {list(cor_cabelo)}", (320, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 80

    # Eye tone
    if 'tom_de_olho' in medidas:
        cv2.putText(painel, "Tom dos Olhos:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cor_olho = tuple(map(int, medidas['tom_de_olho']))
        cv2.rectangle(painel, (200, y_pos - 20), (300, y_pos + 20), cor_olho, -1)
        cv2.putText(painel, f"BGR: {list(cor_olho)}", (320, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 80

    # Classification
    if 'Classificação' in medidas:
        texto = f"Contraste: {medidas['Classificação'].capitalize()}"
        painel = desenhar_texto_com_acentos(painel, texto, (20, y_pos))
        y_pos += 40

    if 'Subtom' in medidas:
        cv2.putText(painel, f"Subtom: {medidas['Subtom'].capitalize()}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return painel


def criar_visualizacoes(imagem, medidas, resultado=None):
    """Create visualizations replacing cv2.imshow()"""
    visualizacoes = {}

    # 1. Original image
    visualizacoes['original'] = imagem.copy()

    # 2. Body analysis (equivalent to your visualizar_resultados)
    if resultado and hasattr(resultado, 'pose_landmarks') and resultado.pose_landmarks:
        try:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose

            imagem_landmarks = imagem.copy()
            mp_drawing.draw_landmarks(
                imagem_landmarks,
                resultado.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            visualizacoes['landmarks'] = imagem_landmarks
        except ImportError:
            st.warning("MediaPipe não disponível para visualização de landmarks")

    # 3. Color panel
    painel_cores = criar_painel_cores(medidas)
    visualizacoes['cores'] = painel_cores

    return visualizacoes


def gerar_recomendacoes_web(dicionario):
    """Web version of clothing recommendation function"""
    try:
        # Try multiple paths for the CSV file
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '../../trabalho/data/catalogo_roupas.csv'),
            os.path.join(os.getcwd(), 'data', 'catalogo_roupas.csv'),
            os.path.join(os.getcwd(), 'catalogo_roupas.csv'),
            'catalogo_roupas.csv'
        ]

        caminho_csv = None
        for path in possible_paths:
            if os.path.exists(path):
                caminho_csv = path
                break

        if not caminho_csv:
            st.error(
                "❌ Arquivo CSV do catálogo não encontrado. Certifique-se de que 'catalogo_roupas.csv' está no diretório correto.")
            return []

        catalogo = pd.read_csv(caminho_csv)
        catalogo.columns = catalogo.columns.str.strip().str.lower()

        # Create copy of catalog
        roupas_filtradas = catalogo.copy()

        # Recommendation rules
        classificacao = dicionario.get('Classificação', '').lower()
        subtom = dicionario.get('Subtom', '').lower()
        contraste = dicionario.get('Classificação', '').lower()
        intensidade = dicionario.get('Intensidade', '').lower()
        profundidade = dicionario.get('Profundidade', '').lower()
        estacao = ''

        if 'estação' in roupas_filtradas.columns:
            if subtom == "quente":
                if intensidade == "alta" and profundidade == "claro":
                        roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("primavera brilhante", case=False, na=False)]
                        estacao = 'primavera brilhante'

                elif intensidade == "baixa":
                    if profundidade == "escuro":
                        roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("outono suave", case=False)]
                        estacao = 'outono suave'
                    else:
                        roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("primavera pura", case=False)]
                        estacao = 'primavera pura'


                elif intensidade == "média":
                    if profundidade == "claro":
                        roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("primavera clara", case=False)]
                        estacao = 'primavera clara'
                    else:
                        roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("outono puro", case=False)]
                        estacao = 'outono puro'

            elif subtom == "frio":
                if intensidade == "alta" and (contraste == "médio contraste" or "baixo contraste escuro"):
                    roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("inverno brilhante", case=False)]
                    estacao = 'inverno brilhante'
                elif intensidade == "baixa":
                    if profundidade == "claro":
                        roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("verão suave", case=False)]
                        estacao = 'verão suave'

                    else:
                        roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("inverno profundo", case=False)]
                        estacao = 'inverno profundo'

                elif intensidade == 'média':
                    if profundidade == "claro":
                        roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("verão claro", case=False)]
                        estacao = 'verão claro'

                    else:
                        roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("inverno puro", case=False)]
                        estacao = 'inverno puro'

            elif subtom == "neutro":
                if profundidade == "claro":
                    roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("verão suave", case=False)]
                    estacao = 'verão suave'

                else:
                    roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("outono suave", case=False)]
                    estacao = 'outono suave'

            elif subtom == "oliva":
                if profundidade == "claro":
                    roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("primavera pura", case=False)]
                    estacao = 'primavera pura'
                else:
                    roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("outono profundo", case=False)]
                    estacao = 'outono profundo'
            else:
                roupas_filtradas = f'Subtom:{subtom}, profundidade: {profundidade}, intensidade {intensidade}'
                estacao = None

        # Convert string "[146 28 63]" to list [146, 28, 63]
        if 'cor bgr' in roupas_filtradas.columns:
            roupas_filtradas["cor bgr"] = roupas_filtradas["cor bgr"].apply(
                lambda x: list(map(int, str(x).strip("[]").split())) if pd.notna(x) else [0, 0, 0]
            )

        # Extract colors
        cores_bgr = []
        for _, row in roupas_filtradas.iterrows():
            if 'cor bgr' in row and isinstance(row['cor bgr'], list) and len(row['cor bgr']) == 3:
                cores_bgr.append(row['cor bgr'])

        return cores_bgr, estacao

    except Exception as e:
        st.error(f"Erro ao processar recomendações: {str(e)}")
        return [], None


def create_color_palette_report(cores_bgr, medidas):
    """Create a downloadable text report of the color analysis using io"""
    report_buffer = io.StringIO()

    # Write header
    report_buffer.write("RELATÓRIO DE ANÁLISE DE COLORAÇÃO PESSOAL\n")
    report_buffer.write("=" * 50 + "\n\n")

    # Write personal analysis
    report_buffer.write("ANÁLISE PESSOAL:\n")
    report_buffer.write("-" * 20 + "\n")

    for key, value in medidas.items():
        if key in ['Classificação', 'Subtom', 'Tom de pele (escala 0-10)', 'Tom de cabelo (escala 0-10)',
                   'Tom dos olhos (escala 0-10)', 'Intensidade']:
            report_buffer.write(f"{key}: {value}\n")

    report_buffer.write("\n")

    # Write recommended colors
    report_buffer.write("CORES RECOMENDADAS (RGB):\n")
    report_buffer.write("-" * 30 + "\n")

    for i, cor_bgr in enumerate(cores_bgr, 1):
        cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0])  # Convert BGR to RGB
        cor_hex = f"#{cor_rgb[0]:02x}{cor_rgb[1]:02x}{cor_rgb[2]:02x}"
        report_buffer.write(f"Cor {i:2d}: RGB{cor_rgb} - HEX: {cor_hex}\n")

    report_buffer.write(f"\nTotal de cores recomendadas: {len(cores_bgr)}\n")
    report_buffer.write("\nRelatório gerado automaticamente pelo sistema de análise de coloração pessoal.\n")

    # Get the content and close the buffer
    content = report_buffer.getvalue()
    report_buffer.close()

    return content


def display_color_grid(cores_bgr):
    """Display colors in a grid format"""
    if not cores_bgr:
        st.warning("Nenhuma cor encontrada para exibir.")
        return

    # Create color grid
    cols_per_row = 5
    rows = len(cores_bgr) // cols_per_row + (1 if len(cores_bgr) % cols_per_row else 0)

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            color_idx = row * cols_per_row + col_idx
            if color_idx < len(cores_bgr):
                with cols[col_idx]:
                    cor_bgr = cores_bgr[color_idx]
                    # Convert BGR to RGB for correct display
                    cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0])
                    cor_hex = f"#{cor_rgb[0]:02x}{cor_rgb[1]:02x}{cor_rgb[2]:02x}"

                    # Create colored square
                    st.markdown(f"""
                    <div style="
                        width: 80px; 
                        height: 80px; 
                        background-color: {cor_hex}; 
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        margin: 5px auto;
                    "></div>
                    <p style="text-align: center; font-size: 10px; margin: 0;">
                        RGB: {cor_rgb}
                    </p>
                    """, unsafe_allow_html=True)


def exibir_imagens_roupas(caminho_imagens):
    """
    Função para ler e exibir imagens de roupas no Streamlit

    Args:
        caminho_imagens (str): Caminho para o diretório das imagens
    """

    # Verificar se o diretório existe
    if not os.path.exists(caminho_imagens):
        st.error(f"Diretório não encontrado: {caminho_imagens}")
        return

    # Extensões de imagem suportadas
    extensoes = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

    # Buscar todas as imagens no diretório
    imagens = []
    try:
        # Listar todos os arquivos no diretório
        arquivos = os.listdir(caminho_imagens)

        for arquivo in arquivos:
            # Verificar se o arquivo tem uma extensão de imagem válida
            nome, ext = os.path.splitext(arquivo.lower())
            if ext in extensoes:
                caminho_completo = os.path.join(caminho_imagens, arquivo)
                imagens.append(caminho_completo)

        # Remover duplicatas (caso existam)
        imagens = list(set(imagens))
        # Ordenar para manter consistência
        imagens.sort()

    except PermissionError:
        st.error("Sem permissão para acessar o diretório.")
        return
    except Exception as e:
        st.error(f"Erro ao acessar o diretório: {str(e)}")
        return

    if not imagens:
        st.warning("Nenhuma imagem encontrada no diretório especificado.")
        return

    # Configurações fixas
    num_colunas = 4
    largura_imagem = 200

    # Criar layout de colunas para as imagens
    colunas = st.columns(num_colunas)

    # Exibir as imagens
    for i, caminho_imagem in enumerate(imagens):
        try:
            # Abrir a imagem
            img = Image.open(caminho_imagem)

            # Calcular coluna atual
            col_atual = i % num_colunas

            with colunas[col_atual]:
                # Exibir a imagem
                st.image(img, width=largura_imagem, use_container_width=False)

        except Exception as e:
            st.error(f"Erro ao carregar imagem {os.path.basename(caminho_imagem)}: {str(e)}")

def substituir_roupas(image):
    client = Client("yisol/IDM-VTON")

    # Upload da roupa principal (garm_img)
    uploaded_roupa_img = st.file_uploader("Arraste a imagem da roupa ou envie alguma de sua escolha:", type=["jpg", "png", "jpeg", "webp"])

    if image and uploaded_roupa_img:
        st.write("Preparando imagens para o modelo...")

        try:
            # Salva imagem original do usuário (modelo)
            temp_model_path = os.path.join(tempfile.gettempdir(), "model_image.png")
            image.save(temp_model_path)

            # Salva imagem da roupa
            temp_roupa_path = os.path.join(tempfile.gettempdir(), "roupa_image.png")
            with open(temp_roupa_path, "wb") as f:
                f.write(uploaded_roupa_img.read())

            # Indica ao usuário que o processamento está em andamento
            with st.spinner("Gerando imagem com nova roupa..."):
                result = client.predict(
                    dict={
                        "background": file(temp_model_path),
                        "layers": [],
                        "composite": None
                    },
                    garm_img=file(temp_roupa_path),
                    garment_des="Roupa enviada pelo usuário",
                    is_checked=True,
                    is_checked_crop=False,
                    denoise_steps=30,
                    seed=42,
                    api_name="/tryon"
                )

            # Garante que o resultado tenha dois caminhos válidos
            if not isinstance(result, (list, tuple)) or len(result) < 1:
                st.error("Erro: resposta inesperada do modelo.")
                return

            output_path = result[0]

            # Copia a imagem de saída
            temp_out_path = os.path.join(tempfile.gettempdir(), "resultado_final.png")
            shutil.copy(output_path, temp_out_path)

            # Exibe e permite download
            st.image(temp_out_path, caption="Resultado")
            with open(temp_out_path, "rb") as f:
                st.download_button("Baixar imagem gerada", f, file_name="resultado.png")

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar a imagem: {e}")

# def substituir_roupas_2(image):
#     # Upload da roupa
#     uploaded_roupa_img = st.file_uploader("Arraste a imagem da roupa ou envie alguma de sua escolha:", type=["jpg", "png", "jpeg", "webp"])

#     if image and uploaded_roupa_img:
#         st.write("Preparando imagens para envio à API...")

#         try:
#             # Salva imagem do modelo (pessoa)
#             temp_model_path = os.path.join(tempfile.gettempdir(), "model_image.jpg")
#             image.save(temp_model_path)

#             # Salva imagem da roupa
#             temp_roupa_path = os.path.join(tempfile.gettempdir(), "roupa_image.jpg")
#             with open(temp_roupa_path, "wb") as f:
#                 f.write(uploaded_roupa_img.read())

#             # Monta o payload
#             with open(temp_model_path, "rb") as model_file, open(temp_roupa_path, "rb") as roupa_file:
#                 files = {
#                     "image": model_file,
#                     "cloth": roupa_file
#                 }

#                 headers = {
#                     "x-rapidapi-key": "cbdf7f1abcmshbdd93e49bcc8466p132ccdjsnfe426af1b090",
#                     "x-rapidapi-host": "try-on-diffusion.p.rapidapi.com"
#                 }

#                 with st.spinner("Processando imagem..."):
#                     response = requests.post(
#                         "https://try-on-diffusion.p.rapidapi.com/try-on-file",
#                         files=files,
#                         headers=headers
#                     )

#             if response.status_code != 200:
#                 st.error(f"Erro da API: {response.status_code} - {response.text}")
#                 return

#             # A resposta da API vem com a URL da imagem resultante
#             result_json = response.json()
#             output_url = result_json.get("output_url")

#             if output_url:
#                 st.image(output_url, caption="Resultado")
#                 st.markdown(f"[Clique aqui para baixar a imagem gerada]({output_url})", unsafe_allow_html=True)
#             else:
#                 st.error("A resposta não contém uma imagem válida.")

#         except Exception as e:
#             st.error(f"Ocorreu um erro: {e}")

def substituir_roupas_2(image):
    """
    Função para substituir roupas em uma imagem usando API de try-on
    
    Args:
        image: Imagem PIL da pessoa/modelo
    """
    # Upload da roupa
    uploaded_roupa_img = st.file_uploader(
        "Arraste a imagem da roupa ou envie alguma de sua escolha:", 
        type=["jpg", "png", "jpeg", "webp"]
    )
    
    if not image:
        st.warning("Por favor, forneça uma imagem do modelo.")
        return
    
    if not uploaded_roupa_img:
        st.info("Aguardando upload da imagem da roupa...")
        return
    
    st.write("Preparando imagens para envio à API...")
    
    temp_model_path = None
    temp_roupa_path = None
    
    try:
        # Validação do tipo de imagem do modelo
        if not hasattr(image, 'save'):
            st.error("Formato de imagem do modelo inválido.")
            return
        
        # Validação do arquivo de roupa
        if uploaded_roupa_img.size == 0:
            st.error("Arquivo da roupa está vazio.")
            return
        
        # Cria diretório temporário se não existir
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        
        # Salva imagem do modelo (pessoa)
        temp_model_path = os.path.join(temp_dir, f"model_image_{int(time.time())}.jpg")
        try:
            # Redimensiona a imagem se necessário (algumas APIs têm limites)
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            image.save(temp_model_path, format='JPEG', quality=95)
            st.info(f"Imagem do modelo salva: {image.size} pixels")
        except Exception as e:
            st.error(f"Erro ao salvar imagem do modelo: {str(e)}")
            return
        
        # Verifica se o arquivo do modelo foi salvo corretamente
        if not os.path.exists(temp_model_path) or os.path.getsize(temp_model_path) == 0:
            st.error("Falha ao salvar imagem do modelo.")
            return
        
        # Salva imagem da roupa
        temp_roupa_path = os.path.join(temp_dir, f"roupa_image_{int(time.time())}.jpg")
        try:
            # Reset do ponteiro do arquivo
            uploaded_roupa_img.seek(0)
            roupa_bytes = uploaded_roupa_img.read()
            
            if len(roupa_bytes) == 0:
                st.error("Arquivo da roupa está vazio ou corrompido.")
                return
            
            # Processa a imagem da roupa para garantir formato correto
            from PIL import Image
            import io
            
            roupa_image = Image.open(io.BytesIO(roupa_bytes))
            # Redimensiona se necessário
            if roupa_image.size[0] > 1024 or roupa_image.size[1] > 1024:
                roupa_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Converte para RGB se necessário
            if roupa_image.mode != 'RGB':
                roupa_image = roupa_image.convert('RGB')
            
            roupa_image.save(temp_roupa_path, format='JPEG', quality=95)
            st.info(f"Imagem da roupa salva: {roupa_image.size} pixels")
            
        except Exception as e:
            st.error(f"Erro ao salvar imagem da roupa: {str(e)}")
            return
        
        # Verifica se o arquivo da roupa foi salvo corretamente
        if not os.path.exists(temp_roupa_path) or os.path.getsize(temp_roupa_path) == 0:
            st.error("Falha ao salvar imagem da roupa.")
            return
        
        # Configuração da API
        headers = {
            "x-rapidapi-key": "cbdf7f1abcmshbdd93e49bcc8466p132ccdjsnfe426af1b090",
            "x-rapidapi-host": "try-on-diffusion.p.rapidapi.com"
        }
        
        # Debug: Mostra informações das imagens antes do envio
        st.info("Enviando para API...")
        st.info(f"Modelo: {os.path.getsize(temp_model_path)} bytes")
        st.info(f"Roupa: {os.path.getsize(temp_roupa_path)} bytes")
        
        # Mostra preview das imagens que serão enviadas
        col1, col2 = st.columns(2)
        with col1:
            st.image(temp_model_path, caption="Modelo (enviado)", width=200)
        with col2:
            st.image(temp_roupa_path, caption="Roupa (enviada)", width=200)
        
        # Faz a requisição para a API
        with st.spinner("Processando imagem..."):
            try:
                with open(temp_model_path, "rb") as model_file, open(temp_roupa_path, "rb") as roupa_file:
                    files = {
                        "image": model_file,
                        "cloth": roupa_file
                    }
                    
                    response = requests.post(
                        "https://try-on-diffusion.p.rapidapi.com/try-on-file",
                        files=files,
                        headers=headers,
                        timeout=60  # Timeout de 60 segundos
                    )
                    
            except requests.exceptions.Timeout:
                st.error("Timeout: A API demorou muito para responder. Tente novamente.")
                return
            except requests.exceptions.ConnectionError:
                st.error("Erro de conexão: Verifique sua conexão com a internet.")
                return
            except requests.exceptions.RequestException as e:
                st.error(f"Erro na requisição: {str(e)}")
                return
        
        # Verifica o status da resposta
        if response.status_code == 200:
            # Verifica o tipo de conteúdo da resposta
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                # Resposta é JSON com URL da imagem
                try:
                    result_json = response.json()
                    output_url = result_json.get("output_url")
                    if output_url:
                        try:
                            # Verifica se a URL é válida
                            url_response = requests.head(output_url, timeout=10)
                            if url_response.status_code == 200:
                                st.success("Imagem processada com sucesso!")
                                st.image(output_url, caption="Resultado")
                                st.markdown(
                                    f"[Clique aqui para baixar a imagem gerada]({output_url})", 
                                    unsafe_allow_html=True
                                )
                            else:
                                st.error("URL da imagem gerada não está acessível.")
                        except Exception as e:
                            st.warning(f"Não foi possível verificar a URL, mas exibindo resultado: {str(e)}")
                            st.image(output_url, caption="Resultado")
                            st.markdown(
                                f"[Clique aqui para baixar a imagem gerada]({output_url})", 
                                unsafe_allow_html=True
                            )
                    else:
                        st.error("A resposta da API não contém uma URL válida para a imagem.")
                        st.error(f"Resposta recebida: {result_json}")
                except ValueError as e:
                    st.error(f"Erro ao decodificar resposta JSON: {str(e)}")
                    return
            
            elif any(img_type in content_type for img_type in ['image/jpeg', 'image/png', 'image/webp', 'image/']):
                # Resposta é uma imagem diretamente
                try:
                    # Salva a imagem recebida
                    result_image_path = os.path.join(temp_dir, f"resultado_{int(time.time())}.jpg")
                    with open(result_image_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Verifica se o arquivo foi salvo corretamente
                    if os.path.exists(result_image_path) and os.path.getsize(result_image_path) > 0:
                        st.success("Imagem processada com sucesso!")
                        st.image(result_image_path, caption="Resultado")
                        
                        # Oferece download da imagem
                        with open(result_image_path, 'rb') as f:
                            st.download_button(
                                label="Baixar imagem gerada",
                                data=f.read(),
                                file_name=f"resultado_try_on_{int(time.time())}.jpg",
                                mime="image/jpeg"
                            )
                        
                        # Limpa o arquivo de resultado
                        try:
                            os.remove(result_image_path)
                        except:
                            pass
                    else:
                        st.error("Falha ao salvar imagem resultado.")
                        
                except Exception as e:
                    st.error(f"Erro ao processar imagem resultado: {str(e)}")
                    return
            
            else:
                # Tipo de conteúdo desconhecido
                st.error(f"Tipo de resposta inesperado: {content_type}")
                st.error(f"Primeiros 200 caracteres da resposta: {str(response.content[:200])}")
                return
        
        elif response.status_code == 400:
            st.error("Erro 400: Dados inválidos enviados para a API. Verifique as imagens.")
        elif response.status_code == 401:
            st.error("Erro 401: Chave da API inválida ou não autorizada.")
        elif response.status_code == 429:
            st.error("Erro 429: Muitas requisições. Tente novamente em alguns minutos.")
        elif response.status_code == 500:
            st.error("Erro 500: Erro interno da API. Tente novamente mais tarde.")
        else:
            st.error(f"Erro da API: {response.status_code}")
            try:
                error_response = response.json()
                st.error(f"Detalhes: {error_response}")
            except:
                st.error(f"Resposta: {response.text[:500]}...")
    
    except Exception as e:
        st.error(f"Erro inesperado: {str(e)}")
        st.error("Tente novamente ou entre em contato com o suporte.")
    
    finally:
        # Limpeza dos arquivos temporários
        try:
            if temp_model_path and os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            if temp_roupa_path and os.path.exists(temp_roupa_path):
                os.remove(temp_roupa_path)
        except Exception as e:
            st.warning(f"Aviso: Não foi possível limpar arquivos temporários: {str(e)}")

def substituir_roupas_3(person_image):
    """
    Função para substituir a roupa em uma imagem de uma pessoa usando o modelo Leffa VTON.

    Esta função cria uma interface no Streamlit para que o usuário envie uma imagem de roupa,
    ajuste os parâmetros do modelo e gere o resultado.

    Args:
        person_image (PIL.Image.Image): A imagem da pessoa (modelo) carregada
                                         anteriormente na aplicação Streamlit.
    """
    try:
        # Inicializa o cliente para a API do Gradio
        client = Client("franciszzj/Leffa")
    except Exception as e:
        st.error(f"Não foi possível conectar ao cliente da API do Gradio: {e}")
        return

    # --- UI do Streamlit para Opções do Modelo ---
    st.sidebar.header("Opções de Geração (Leffa VTON)")
    
    vt_garment_type = st.sidebar.radio(
        "Tipo de Peça de Roupa",
        options=['upper_body', 'lower_body', 'dresses'],
        captions=["Parte de cima (camisetas, blusas)", "Parte de baixo (calças, saias)", "Vestidos"],
        index=0
    )

    vt_model_type = st.sidebar.radio(
        "Tipo do Modelo de IA",
        options=['viton_hd', 'dress_code'],
        index=0
    )

    step = st.sidebar.slider("Passos de Inferência (Steps)", min_value=10, max_value=100, value=30, step=1)
    scale = st.sidebar.slider("Escala de Orientação (Guidance Scale)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    seed = st.sidebar.number_input("Semente Aleatória (Seed)", value=42)

    # --- Upload da Imagem da Roupa ---
    uploaded_garment_img = st.file_uploader(
        "Arraste a imagem da peça de roupa:",
        type=["jpg", "png", "jpeg", "webp"]
    )

    # Verifica se as duas imagens (pessoa e roupa) foram fornecidas
    if person_image and uploaded_garment_img:
        
        # Botão para iniciar o processamento
        if st.button("✨ Aplicar Nova Roupa"):
            st.write("Preparando imagens para o modelo...")
            temp_dir = None # Inicializa a variável para o bloco finally
            try:
                # Cria diretórios temporários para salvar as imagens
                temp_dir = tempfile.mkdtemp()
                temp_person_path = os.path.join(temp_dir, "person_image.png")
                temp_garment_path = os.path.join(temp_dir, "garment_image.png")

                # Salva a imagem da pessoa (modelo) no caminho temporário
                person_image.save(temp_person_path)

                # Salva a imagem da roupa no caminho temporário
                with open(temp_garment_path, "wb") as f:
                    f.write(uploaded_garment_img.read())

                # Mostra um spinner enquanto o modelo processa
                with st.spinner("Gerando imagem com a nova roupa... Isso pode levar um momento."):
                    result = client.predict(
                        src_image_path=handle_file(temp_person_path),
                        ref_image_path=handle_file(temp_garment_path),
                        ref_acceleration=False,
                        step=float(step),
                        scale=float(scale),
                        seed=float(seed),
                        vt_model_type=vt_model_type,
                        vt_garment_type=vt_garment_type,
                        vt_repaint=False,
                        api_name="/leffa_predict_vt"
                    )

                # A API retorna uma tupla. A imagem gerada é o primeiro item.
                if not isinstance(result, (list, tuple)) or len(result) < 1:
                    st.error("Erro: A resposta do modelo foi inesperada.")
                    return

                # O resultado é um dicionário contendo o caminho do arquivo de saída
                output_file_info = result[0]
                if 'path' not in output_file_info:
                    st.error("Erro: O caminho do arquivo de resultado não foi encontrado na resposta da API.")
                    return
                
                output_path = output_file_info['path']

                # Copia a imagem de saída para um local temporário seguro para exibição
                final_image_path = os.path.join(temp_dir, "resultado_final.png")
                shutil.copy(output_path, final_image_path)

                # Exibe a imagem resultante
                st.image(final_image_path, caption="Resultado da Troca de Roupa")

                # Cria um botão de download para a imagem gerada
                with open(final_image_path, "rb") as f:
                    st.download_button(
                        "Baixar Imagem Gerada",
                        data=f,
                        file_name="resultado_troca_roupa.png",
                        mime="image/png"
                    )

            except Exception as e:
                error_message = str(e)
                if "The upstream Gradio app has raised an exception" in error_message:
                    st.error(
                        "**Ocorreu um erro no servidor do modelo de IA.**\n\n"
                        "Isso geralmente acontece por um dos seguintes motivos:\n\n"
                        "1.  **Imagem Inválida:** Tente usar uma imagem de pessoa ou de roupa diferente. Imagens com dimensões muito grandes, transparência (PNG) ou formatos incomuns podem causar falhas.\n"
                        "2.  **Serviço Indisponível:** O serviço no Hugging Face pode estar temporariamente sobrecarregado ou offline. Por favor, tente novamente mais tarde."
                    )
                else:
                    st.error(f"Ocorreu um erro inesperado durante o processamento: {error_message}")
            finally:
                # Limpa o diretório temporário após o uso
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

def substituir_roupas_4(image_modelo):
    """
    Função para um provador virtual que utiliza o modelo jallenjia/Change-Clothes-AI.

    Args:
        image_modelo: Uma imagem (do tipo PIL Image) da pessoa que vai provar a roupa.
    """
    st.header("Passo 2: Envie a peça de roupa")

    # Inicializa o cliente da API do Gradio
    try:
        client = Client("jallenjia/Change-Clothes-AI")
    except Exception as e:
        st.error(f"Não foi possível conectar ao serviço de IA: {e}")
        return

    # --- Inputs do Usuário ---
    # 1. Upload da imagem da roupa
    uploaded_roupa_img = st.file_uploader(
        "Arraste a imagem da roupa ou envie alguma de sua escolha:",
        type=["jpg", "png", "jpeg", "webp"]
    )

    # 2. Seleção da categoria da roupa
    # categoria_roupa = st.selectbox(
    #     "Selecione a categoria da peça:",
    #     ('upper_body', 'lower_body', 'dresses'),
    #     help="Escolha 'upper_body' para camisetas e blusas, 'lower_body' para calças e saias, e 'dresses' para vestidos."
    # )
    categoria_roupa = 'upper_body'
    
    # 3. Descrição opcional da roupa
    # descricao_roupa = st.text_input(
    #     "Descrição da roupa (opcional):", 
    #     "uma peça de roupa",
    #     help="Ex: 'uma camisa de algodão azul'"
    # )
    descricao_roupa = ''

    # --- Processamento ---
    if image_modelo and uploaded_roupa_img:
        st.write("Preparando imagens para o modelo...")

        try:
            # Salva a imagem da pessoa (modelo) em um arquivo temporário
            temp_modelo_path = os.path.join(tempfile.gettempdir(), "modelo_image.png")
            image_modelo.save(temp_modelo_path)

            # Salva a imagem da roupa em um arquivo temporário
            temp_roupa_path = os.path.join(tempfile.gettempdir(), "roupa_image.png")
            with open(temp_roupa_path, "wb") as f:
                f.write(uploaded_roupa_img.getvalue())

            # Indica ao usuário que o processamento está em andamento
            with st.spinner("Aplicando a roupa na imagem... Isso pode levar um minuto."):
                result = client.predict(
                    # Parâmetro 1: Dicionário com a imagem de fundo (a pessoa)
                    dict={
                        "background": handle_file(temp_modelo_path),
                        "layers": [],
                        "composite": None
                    },
                    # Parâmetro 2: Imagem da peça de roupa
                    garm_img=handle_file(temp_roupa_path),
                    # Parâmetro 3: Descrição da roupa
                    garment_des=descricao_roupa,
                    # Parâmetro 4: Usar auto-masking (padrão)
                    is_checked=True,
                    # Parâmetro 5: Não cortar a imagem da roupa (padrão)
                    is_checked_crop=False,
                    # Parâmetro 6: Passos de remoção de ruído (qualidade)
                    denoise_steps=30,
                    # Parâmetro 7: Semente para aleatoriedade (-1 para aleatório)
                    seed=-1,
                    # Parâmetro 8: Categoria da roupa (essencial para o modelo)
                    category=categoria_roupa,
                    # Endpoint da API
                    api_name="/tryon"
                )

            # A API retorna uma tupla com 2 caminhos de arquivo
            if not isinstance(result, (list, tuple)) or len(result) < 2:
                st.error("Erro: A resposta do modelo foi inesperada. Tente novamente.")
                st.write("Resposta recebida:", result) # Para depuração
                return

            # O primeiro item é a imagem final
            output_path = result[0]
            
            # O segundo item (opcional) é a máscara, podemos ignorar ou exibir
            # masked_path = result[1] 

            # Copia a imagem de saída para um local temporário seguro
            temp_out_path = os.path.join(tempfile.gettempdir(), "resultado_final.png")
            shutil.copy(output_path, temp_out_path)

            # Exibe o resultado e o botão de download
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(temp_out_path, caption="✨ Imagem Finalizada! ✨", use_container_width=True)
                with open(temp_out_path, "rb") as f:
                    st.download_button(
                        "Baixar imagem gerada", 
                        f, 
                        file_name="look_virtual.png",
                        mime="image/png"
                    )

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar a imagem: {e}")
        finally:
            # Limpa os arquivos temporários para não ocupar espaço
            if 'temp_modelo_path' in locals() and os.path.exists(temp_modelo_path):
                os.remove(temp_modelo_path)
            if 'temp_roupa_path' in locals() and os.path.exists(temp_roupa_path):
                os.remove(temp_roupa_path)

def main():
    # # Header
    # st.markdown('<h1 class="main-header">VesteAI</h1>', unsafe_allow_html=True)

    # Exibe os slides no app
    diretorio_slide = os.path.dirname(os.path.abspath(__file__))
    st.image(Image.open(os.path.join(diretorio_slide, "..", "data", "slides", "slide1.png")), use_container_width=True)
    st.image(Image.open(os.path.join(diretorio_slide, "..", "data", "slides", "slide2.png")), use_container_width=True)
    st.image(Image.open(os.path.join(diretorio_slide, "..", "data", "slides", "slide3.png")), use_container_width=True)
    st.image(Image.open(os.path.join(diretorio_slide, "..", "data", "slides", "slide6.png")), use_container_width=True)
    st.image(Image.open(os.path.join(diretorio_slide, "..", "data", "slides", "slide4.png")), use_container_width=True)

    st.markdown('<h1 class="main-header">🎨 Análise da Coloração Pessoal</h1>', unsafe_allow_html=True)
    st.markdown("**Upload uma foto para análise completa das suas características de cor e estilo!**")

    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 Instruções")
        st.markdown("""
        **Como funciona?**
        1.  **Envie sua foto:** Use o botão ao lado.
        2.  **Aguarde a análise:** Nossa IA processará sua imagem.
        3.  **Receba sua paleta:** Descubra sua estação e cores ideais!
        
        **Dicas para a foto:**
        - Foto de corpo inteiro,
        - Iluminação natural,
        - Rosto bem visível,
        - Fundo neutro,
        - Evite filtros ou edições na imagem.
        """)

        st.header("ℹ️ Sobre a Análise")
        st.markdown("""
        Esta ferramenta analisa:
        - Tom de pele, cabelo e olhos
        - Contraste facial
        - Subtom (quente/frio/neutro/oliva)
        - Recomendações de cores
        - Sugestão de roupas
        """)
        st.markdown("---")
        st.caption("Desenvolvido com ❤️ e IA.")

    uploaded_file = st.file_uploader(
        "Escolha uma imagem",
        type=['png', 'jpg', 'jpeg', 'webp'],        
        help="Faça upload de uma foto com boa iluminação"
    )

    if uploaded_file is not None:
        # Show uploaded image
        image_bytes = uploaded_file.read()
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📸 Imagem Enviada")
            st.image(image, caption="Sua foto", use_container_width=True)

        with col2:
            st.subheader("📊 Resultados da Análise")

            if not PROCESSAMENTO_AVAILABLE:
                st.error(
                    "Módulo de processamento não disponível. Verifique se o arquivo 'processamento.py' está presente.")
                st.stop()  # Use stop() em vez de return para interromper a execução

            # Executa a análise automaticamente
            with st.spinner("Analisando sua coloração pessoal..."):
                try:
                    # Convert PIL to OpenCV
                    cv_image = pil_to_opencv(image)

                    # Call your analysis function
                    medidas, resultado = extrair_dados_da_imagem(cv_image, image_bytes)

                    # Create visualizations
                    visualizacoes = criar_visualizacoes(cv_image, medidas, resultado)

                    # Store in session_state
                    st.session_state.medidas = medidas
                    st.session_state.visualizacoes = visualizacoes
                    st.session_state.analysis_complete = True

                    face_landmarks = None
                
                except Exception as e:
                    st.error(f"Erro na análise: {str(e)}")
                    st.code(traceback.format_exc())


            # Display results if they exist - MOVED INSIDE COL2
            if st.session_state.get('analysis_complete', False) and 'medidas' in st.session_state:

                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    st.markdown("### 🧍 Medidas Corporais")
                    medidas_corporais = {
                        k: v for k, v in st.session_state.medidas.items()
                        if k in ['altura_total', 'largura_ombros', 'largura_quadril', 'proporção',
                                 'Tipo de corpo', 'Formato do rosto']
                    }
                    if medidas_corporais:
                        for key, value in medidas_corporais.items():
                            st.metric(key.replace('_', ' ').title(), value)
                    else:
                        st.info("Medidas corporais não detectadas")

                with col_res2:
                    st.markdown("### 🎨 Análise de Cores")
                    analise_cores = {
                        k: v for k, v in st.session_state.medidas.items()
                        if k in ['Classificação', 'Subtom', 'Tom de pele (escala 0-10)',
                                 'Tom de cabelo (escala 0-10)', 'Tom dos olhos (escala 0-10)', 'Intensidade']
                    }
                    if analise_cores:
                        for key, value in analise_cores.items():
                            st.metric(key, value)
                    else:
                        st.info("Análise de cores não disponível")

        # Section 3: Clothing recommendations
        st.divider()
        st.subheader("👗 Recomendações de Cores")

        with st.spinner("Buscando roupas ideais para você..."):
            try:
                cores_recomendadas, estacao = gerar_recomendacoes_web(st.session_state.medidas)

                if cores_recomendadas:
                    st.subheader(f"🎨PARABÉNS! A sua estação é {estacao.capitalize()}")
                    display_color_grid(cores_recomendadas)

                    # Create downloadable color palette
                    try:
                        palette_data = create_color_palette_report(cores_recomendadas, st.session_state.medidas)
                        st.download_button(
                            label="📥 Baixar Relatório de Cores",
                            data=palette_data,
                            file_name="color_palette_report.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Erro ao criar relatório: {e}")

                    st.write("Lembre-se: este é um guia. Sinta-se livre para experimentar e usar o que te faz sentir bem!")
                else:
                    resultado = gerar_recomendacoes_web(st.session_state.medidas)
                    st.write("Resultado da função:", resultado)
                    st.warning(
                        "⚠️ Nenhuma roupa recomendada encontrada. Verifique se o arquivo CSV do catálogo está disponível.")

            except Exception as e:
                st.error(f"Erro nas recomendações: {str(e)}")
                st.code(traceback.format_exc())

        # Section 4: Clothing recommendations
        st.divider()
        
        diretorio_roupas = os.path.dirname(os.path.abspath(__file__))
        
        st.subheader(f"🧥 Roupas para corpo {st.session_state.medidas['Tipo de corpo'].lower()}")
        st.subheader("OBJETIVO:")
        if st.session_state.medidas['Tipo de corpo'] == 'Triângulo Invertido':
            st.markdown('**Suavizar os ombros e dar mais equilíbrio ou volume visual à parte inferior.**')
            st.subheader("🔷 MASCULINO:")
            col_utilize1, col_evite1 = st.columns([1, 1])
            col_utilize2, col_evite2 = st.columns([1, 1])
            with col_utilize1:
                st.subheader("UTILIZE:")
                st.markdown("""
                        **ROUPAS QUE DIMINUEM VISUALMENTE A PARTE SUPERIOR**
                        - Camiseta gola V com tecido leve
                        - Camisa de algodão com caimento reto
                        - Jaqueta estilo bomber ou reta
                        """)

            with col_evite1:
                st.subheader("EVITE: ")
                st.markdown("""
                            **ROUPAS QUE ADICIONEM VOLUME À PARTE SUPERIOR**
                            - Roupas com ombreiras
                            - Babados ou brilhos na região dos ombros
                            """)
            with col_utilize2:
                st.markdown("""
                        **ROUPAS QUE AUMENTEM VISUALMENTE A PARTE INFERIOR**
                        - Calça cargo (com bolsos laterais) / Bermuda cargo
                        - Calça reta com tecido estruturado (como sarja)
                        - Calça com punho na barra (tipo jogger mais larga)
                        """)
            with col_evite2:
                st.markdown("""
                            **ROUPAS QUE REDUZAM VISUALMENTE A PARTE INFERIOR**
                            - Leggings e calça skinny, saias justas
                            - Calças sem volume na região do quadril
                            """)
            st.subheader("🔶 FEMININO:")
            col_utilize3, col_evite3 = st.columns([1, 1])
            col_utilize4, col_evite4 = st.columns([1, 1])
            with col_utilize3:
                st.subheader("UTILIZE:")
                st.markdown("""
                        **ROUPAS QUE DIMINUEM VISUALMENTE A PARTE SUPERIOR**
                        - Blusa com decote V ou U
                        - Regata de alças finas
                        - Camisas de tecido fluido (como viscose)
                        """)

            with col_evite3:
                st.subheader("EVITE: ")
                st.markdown("""
                            **ROUPAS QUE ADICIONEM VOLUME À PARTE SUPERIOR**
                            - Roupas com ombreiras
                            - Babados ou brilhos na região dos ombros
                            """)
            with col_utilize4:
                st.markdown("""
                        **ROUPAS QUE AUMENTEM VISUALMENTE A PARTE INFERIOR**
                        - Calça pantalona
                        - Calça com barras amplas ou com volume no tornozelo
                        - Saia evasê (A-line)
                        - Short clochard (com franzido na cintura)
                        - Vestido com saia volumosa e top simples
                        - Vestido com recortes na cintura e saia estruturada
                        """)
            with col_evite4:
                st.markdown("""
                            **ROUPAS QUE REDUZAM VISUALMENTE A PARTE INFERIOR**
                            - Leggings e calça skinny, saias justas
                            - Calças sem volume na região do quadril
                            """)
            
            #exibir_imagens_roupas(os.path.join(diretorio_roupas, "..", "data", "imagens_corpos", "triângulo invertido1"))
            exibir_imagens_roupas(os.path.join(diretorio_roupas, "..", "data", "imagens_roupas1"))

        elif st.session_state.medidas['Tipo de corpo'] == 'Triângulo':
            st.markdown('**Suavizar os ombros e dar mais equilíbrio ou volume visual à parte inferior.**')
            st.subheader("🔷 MASCULINO:")
            col_utilize1, col_evite1 = st.columns([1, 1])
            col_utilize2, col_evite2 = st.columns([1, 1])
            with col_utilize1:
                st.subheader("UTILIZE:")
                st.markdown("""
                        **ROUPAS QUE DIMINUEM VISUALMENTE A PARTE SUPERIOR**
                        - Camiseta gola V com tecido leve
                        - Camisa de algodão com caimento reto
                        - Jaqueta estilo bomber ou reta
                        """)

            with col_evite1:
                st.subheader("EVITE: ")
                st.markdown("""
                            **ROUPAS QUE ADICIONEM VOLUME À PARTE SUPERIOR**
                            - Roupas com ombreiras
                            - Babados ou brilhos na região dos ombros
                            """)
            with col_utilize2:
                st.markdown("""
                        **ROUPAS QUE AUMENTEM VISUALMENTE A PARTE INFERIOR**
                        - Calça cargo (com bolsos laterais) / Bermuda cargo
                        - Calça reta com tecido estruturado (como sarja)
                        - Calça com punho na barra (tipo jogger mais larga)
                        """)
            with col_evite2:
                st.markdown("""
                            **ROUPAS QUE REDUZAM VISUALMENTE A PARTE INFERIOR**
                            - Leggings e calça skinny, saias justas
                            - Calças sem volume na região do quadril
                            """)
            st.subheader("🔶 FEMININO:")
            col_utilize3, col_evite3 = st.columns([1, 1])
            col_utilize4, col_evite4 = st.columns([1, 1])
            with col_utilize3:
                st.subheader("UTILIZE:")
                st.markdown("""
                        **ROUPAS QUE DIMINUEM VISUALMENTE A PARTE SUPERIOR**
                        - Blusa com decote V ou U
                        - Regata de alças finas
                        - Camisas de tecido fluido (como viscose)
                        """)

            with col_evite3:
                st.subheader("EVITE: ")
                st.markdown("""
                            **ROUPAS QUE ADICIONEM VOLUME À PARTE SUPERIOR**
                            - Roupas com ombreiras
                            - Babados ou brilhos na região dos ombros
                            """)
            with col_utilize4:
                st.markdown("""
                        **ROUPAS QUE AUMENTEM VISUALMENTE A PARTE INFERIOR**
                        - Calça pantalona
                        - Calça com barras amplas ou com volume no tornozelo
                        - Saia evasê (A-line)
                        - Short clochard (com franzido na cintura)
                        - Vestido com saia volumosa e top simples
                        - Vestido com recortes na cintura e saia estruturada
                        """)
            with col_evite4:
                st.markdown("""
                            **ROUPAS QUE REDUZAM VISUALMENTE A PARTE INFERIOR**
                            - Leggings e calça skinny, saias justas
                            - Calças sem volume na região do quadril
                            """)

        elif st.session_state.medidas['Tipo de corpo'] == 'Oval':
            st.markdown('**Suavizar os ombros e dar mais equilíbrio ou volume visual à parte inferior.**')
            st.subheader("🔷 MASCULINO:")
            col_utilize1, col_evite1 = st.columns([1, 1])
            col_utilize2, col_evite2 = st.columns([1, 1])
            with col_utilize1:
                st.subheader("UTILIZE:")
                st.markdown("""
                        **ROUPAS QUE DIMINUEM VISUALMENTE A PARTE SUPERIOR**
                        - Camiseta gola V com tecido leve
                        - Camisa de algodão com caimento reto
                        - Jaqueta estilo bomber ou reta
                        """)

            with col_evite1:
                st.subheader("EVITE: ")
                st.markdown("""
                            **ROUPAS QUE ADICIONEM VOLUME À PARTE SUPERIOR**
                            - Roupas com ombreiras
                            - Babados ou brilhos na região dos ombros
                            """)
            with col_utilize2:
                st.markdown("""
                        **ROUPAS QUE AUMENTEM VISUALMENTE A PARTE INFERIOR**
                        - Calça cargo (com bolsos laterais) / Bermuda cargo
                        - Calça reta com tecido estruturado (como sarja)
                        - Calça com punho na barra (tipo jogger mais larga)
                        """)
            with col_evite2:
                st.markdown("""
                            **ROUPAS QUE REDUZAM VISUALMENTE A PARTE INFERIOR**
                            - Leggings e calça skinny, saias justas
                            - Calças sem volume na região do quadril
                            """)
            st.subheader("🔶 FEMININO:")
            col_utilize3, col_evite3 = st.columns([1, 1])
            col_utilize4, col_evite4 = st.columns([1, 1])
            with col_utilize3:
                st.subheader("UTILIZE:")
                st.markdown("""
                        **ROUPAS QUE DIMINUEM VISUALMENTE A PARTE SUPERIOR**
                        - Blusa com decote V ou U
                        - Regata de alças finas
                        - Camisas de tecido fluido (como viscose)
                        """)

            with col_evite3:
                st.subheader("EVITE: ")
                st.markdown("""
                            **ROUPAS QUE ADICIONEM VOLUME À PARTE SUPERIOR**
                            - Roupas com ombreiras
                            - Babados ou brilhos na região dos ombros
                            """)
            with col_utilize4:
                st.markdown("""
                        **ROUPAS QUE AUMENTEM VISUALMENTE A PARTE INFERIOR**
                        - Calça pantalona
                        - Calça com barras amplas ou com volume no tornozelo
                        - Saia evasê (A-line)
                        - Short clochard (com franzido na cintura)
                        - Vestido com saia volumosa e top simples
                        - Vestido com recortes na cintura e saia estruturada
                        """)
            with col_evite4:
                st.markdown("""
                            **ROUPAS QUE REDUZAM VISUALMENTE A PARTE INFERIOR**
                            - Leggings e calça skinny, saias justas
                            - Calças sem volume na região do quadril
                            """)
        elif st.session_state.medidas['Tipo de corpo'] == 'Retângulo (Atlético)':
            st.markdown('**Suavizar os ombros e dar mais equilíbrio ou volume visual à parte inferior.**')
            st.subheader("🔷 MASCULINO:")
            col_utilize1, col_evite1 = st.columns([1, 1])
            col_utilize2, col_evite2 = st.columns([1, 1])
            with col_utilize1:
                st.subheader("UTILIZE:")
                st.markdown("""
                        **ROUPAS QUE DIMINUEM VISUALMENTE A PARTE SUPERIOR**
                        - Camiseta gola V com tecido leve
                        - Camisa de algodão com caimento reto
                        - Jaqueta estilo bomber ou reta
                        """)

            with col_evite1:
                st.subheader("EVITE: ")
                st.markdown("""
                            **ROUPAS QUE ADICIONEM VOLUME À PARTE SUPERIOR**
                            - Roupas com ombreiras
                            - Babados ou brilhos na região dos ombros
                            """)
            with col_utilize2:
                st.markdown("""
                        **ROUPAS QUE AUMENTEM VISUALMENTE A PARTE INFERIOR**
                        - Calça cargo (com bolsos laterais) / Bermuda cargo
                        - Calça reta com tecido estruturado (como sarja)
                        - Calça com punho na barra (tipo jogger mais larga)
                        """)
            with col_evite2:
                st.markdown("""
                            **ROUPAS QUE REDUZAM VISUALMENTE A PARTE INFERIOR**
                            - Leggings e calça skinny, saias justas
                            - Calças sem volume na região do quadril
                            """)
            st.subheader("🔶 FEMININO:")
            col_utilize3, col_evite3 = st.columns([1, 1])
            col_utilize4, col_evite4 = st.columns([1, 1])
            with col_utilize3:
                st.subheader("UTILIZE:")
                st.markdown("""
                        **ROUPAS QUE DIMINUEM VISUALMENTE A PARTE SUPERIOR**
                        - Blusa com decote V ou U
                        - Regata de alças finas
                        - Camisas de tecido fluido (como viscose)
                        """)

            with col_evite3:
                st.subheader("EVITE: ")
                st.markdown("""
                            **ROUPAS QUE ADICIONEM VOLUME À PARTE SUPERIOR**
                            - Roupas com ombreiras
                            - Babados ou brilhos na região dos ombros
                            """)
            with col_utilize4:
                st.markdown("""
                        **ROUPAS QUE AUMENTEM VISUALMENTE A PARTE INFERIOR**
                        - Calça pantalona
                        - Calça com barras amplas ou com volume no tornozelo
                        - Saia evasê (A-line)
                        - Short clochard (com franzido na cintura)
                        - Vestido com saia volumosa e top simples
                        - Vestido com recortes na cintura e saia estruturada
                        """)
            with col_evite4:
                st.markdown("""
                            **ROUPAS QUE REDUZAM VISUALMENTE A PARTE INFERIOR**
                            - Leggings e calça skinny, saias justas
                            - Calças sem volume na região do quadril
                            """)
            # st.title("VesteAI")
            # st.header("Instruções")
            # st.subheader("Como funciona?")
            # st.text("Este é um texto sem formatação.")
            # st.caption("Texto auxiliar em fonte menor.")
            # st.markdown("**Texto em negrito**, _itálico_, e uma [link](https://streamlit.io)")
            # st.code("print('Olá mundo')", language="python")
            # st.latex(r"E = mc^2")

        #substituir_roupas(image)
        #substituir_roupas_2(image)
        substituir_roupas_4(image)

    
        st.image(Image.open(os.path.join(diretorio_slide, "..", "data", "slides", "slide7.png")), use_container_width=True)
        st.image(Image.open(os.path.join(diretorio_slide, "..", "data", "slides", "slide8.png")), use_container_width=True)


        # # Complete dictionary (expandable)
        # with st.expander("📋 Ver Dicionário Completo de Análise"):
        #     st.json(st.session_state.medidas)

if __name__ == "__main__":
    main()
