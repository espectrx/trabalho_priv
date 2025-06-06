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
from gradio_client import Client, file
import shutil
import tempfile

# Try to import your custom functions - with error handling
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
        fonte = ImageFont.truetype("arial.ttf", tamanho)

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
    uploaded_roupa_img = st.file_uploader("Arraste a imagem da roupa ou envie alguma de sua escolha:", type=["jpg", "png", "jpeg"])

    if image and uploaded_roupa_img:
        # Converte a imagem PIL para um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_model:
            image.save(temp_model, format="PNG")
            temp_model_path = temp_model.name

        # Salva a roupa enviada em arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_roupa:
            temp_roupa.write(uploaded_roupa_img.read())
            temp_roupa_path = temp_roupa.name

        # Chama o modelo com os arquivos temporários
        result = client.predict(
            dict={
                "background": file(temp_model_path),
                "layers": [],  # pode adicionar camadas aqui
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

        output_path, masked_path = result

        # Salva o resultado final em outro temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_out:
            shutil.copy(output_path, temp_out.name)
            temp_out_path = temp_out.name

        # Mostra imagem e botão de download
        st.image(temp_out_path, caption="Resultado")
        with open(temp_out_path, "rb") as f:
            st.download_button("Baixar imagem gerada", f, file_name="resultado_vton.png")


def main():
    # Header
    st.markdown('<h1 class="main-header">VesteAI</h1>', unsafe_allow_html=True)
    slide1 = Image.open(r"C:\Users\HOME\Desktop\slide1.png")
    slide2 = Image.open(r"C:\Users\HOME\Desktop\slide2.png")
    slide3 = Image.open(r"C:\Users\HOME\Desktop\slide3.png")
    slide4 = Image.open(r"C:\Users\HOME\Desktop\slide4.png")


    # Exibe a imagem no app
    st.image(slide1, use_container_width=True)
    st.image(slide2, use_container_width=True)
    st.image(slide3, use_container_width=True)
    st.image(slide4, use_container_width=True)

    st.markdown('<h1 class="main-header">🎨 Análise de Coloração Pessoal</h1>', unsafe_allow_html=True)
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
        type=['png', 'jpg', 'jpeg'],
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
            exibir_imagens_roupas(r'C:\Users\HOME\PycharmProjects\trabalhoFinal\trabalho\data\imagens_corpos\triângulo invertido')

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

        substituir_roupas(image)

        # Complete dictionary (expandable)
        with st.expander("📋 Ver Dicionário Completo de Análise"):
            st.json(st.session_state.medidas)

if __name__ == "__main__":
    main()