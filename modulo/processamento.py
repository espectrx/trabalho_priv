import cv2  # manipula imagens_roupas
import mediapipe as mp  # detecta as partes do corpo
import numpy as np
import base64
from PIL import Image
import google.generativeai as genai
from transformers import pipeline, Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, AutoModel
import mediapipe as mp
from collections import Counter
import math
import torch
import json
import time
import os
import requests
import json
from io import BytesIO
import os
import random
import io  # Importar para lidar com bytes como arquivo

medidas = {}
def extrair_dados_da_imagem(imagem,caminho=None):
    # RECEBE UMA IMAGEM BGR E RETORNA UM DICIONÁRIO COM MEDIDAS EXTRAÍDAS DELA
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    # CONVERTE PARA RGB
    img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    # PROCESSA POSE E FACE
    with mp_pose.Pose(static_image_mode=True) as pose, mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
    ) as face_mesh:

        resultado = pose.process(img_rgb)
        resultado_face = face_mesh.process(img_rgb)

    # ================================= CORPO =================================
    if resultado.pose_landmarks:
        # try:
        # ALTURA TOTAL (CABEÇA AO TORNOZELO)
        topo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        tornozelo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        altura_total = round(abs(tornozelo.y - topo.y), 2)
        medidas['altura_total'] = altura_total

        # DISTANCIA OMBROS
        l_shoulder = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        dx = (l_shoulder.x - r_shoulder.x)
        dy = (l_shoulder.y - r_shoulder.y)
        distancia = round(np.sqrt(dx ** 2 + dy ** 2), 2)
        medidas['largura_ombros'] = distancia

        # PROPORÇÃO TRONCO E PERNA
        quadril = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        altura_tronco = abs(l_shoulder.y - quadril.y)
        altura_pernas = abs(quadril.y - tornozelo.y)
        proporcao_tronco_pernas = round(altura_tronco / altura_pernas, 2)
        medidas['proporção'] = proporcao_tronco_pernas

        # LARGURA QUADRIL
        l_hip = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        dx_hip = (l_hip.x - r_hip.x)
        dy_hip = (l_hip.y - r_hip.y)
        largura_quadril = round(np.sqrt(dx_hip ** 2 + dy_hip ** 2), 2)
        medidas['largura_quadril'] = largura_quadril

        # DIFERENÇA OMBRO E QUADRIL
        ombros = medidas.get('largura_ombros')
        quadril = medidas.get('largura_quadril')
        proporcao = medidas.get('proporção')
        diferenca = abs(ombros - quadril)

        # TIPO DE CORPO
        if diferenca < 0.03:
            if proporcao < 0.9:
                tipo_corpo = "Ampulheta"
            else:
                tipo_corpo = "Retângulo"
        elif ombros > quadril:
            tipo_corpo = "Triângulo Invertido"
        elif quadril > ombros:
            tipo_corpo = "Pêra (Triângulo)"
        else:
            tipo_corpo = "Desconhecido"

        medidas['Tipo de corpo'] = tipo_corpo

    #     class MultiGeminiBodyTypeClassifier:
    #         def __init__(self, gemini_api_keys=None):
    #             """
    #             Inicializa o classificador com múltiplas chaves Gemini
    
    #             Args:
    #                 gemini_api_keys: Lista de chaves da API do Gemini ou chave única
    #             """
    
    #             # Configurar múltiplas chaves
    #             if gemini_api_keys is None:
    #                 # Chaves padrão (substitua pelas suas)
    #                 self.api_keys = [
    #                     "AIzaSyBsuDaBYYHhNRLIob8U8Zbb1hKMWAuLASE",  # Chave principal
    #                     "AIzaSyBujAPcUqckJ3vDceiXp2dcjoKSk5tB2jI",  # Chave backup 1
    #                     "AIzaSyBdY1G2LdtQpsw1tAsuyNz5JED5T2gFt5w",  # Chave backup 2
    #                     "AIzaSyApCqbHjrkpVMAMz07HxTKS4Hxas0SAONs"  # Chave backup 3
    #                 ]
    #             elif isinstance(gemini_api_keys, str):
    #                 # Apenas uma chave fornecida
    #                 self.api_keys = [gemini_api_keys]
    #             else:
    #                 # Lista de chaves fornecida
    #                 self.api_keys = gemini_api_keys
    
    #             # Filtrar chaves válidas (remover placeholders)
    #             self.api_keys = [key for key in self.api_keys if not key.startswith("SUA_")]
    
    #             if not self.api_keys:
    #                 raise ValueError("Pelo menos uma chave válida do Gemini deve ser fornecida")
    
    #             print(f"🔑 Configuradas {len(self.api_keys)} chaves Gemini")
    
    #             # Status das chaves (para controle de rate limit)
    #             self.key_status = {key: {"available": True, "last_error": None, "cooldown_until": 0}
    #                                for key in self.api_keys}
    
    #             # Tipos corporais válidos
    #             self.valid_body_types = ['TRIANGULO', 'TRIANGULO_INVERTIDO', 'OVAL', 'RETANGULO']
    
    #         def _get_available_key(self):
    #             """
    #             Retorna uma chave disponível, priorizando as que não estão em cooldown
    
    #             Returns:
    #                 str: Chave API disponível ou None se todas estão indisponíveis
    #             """
    #             current_time = time.time()
    
    #             # Verificar chaves fora de cooldown
    #             available_keys = []
    #             for key in self.api_keys:
    #                 if (self.key_status[key]["available"] and
    #                         current_time > self.key_status[key]["cooldown_until"]):
    #                     available_keys.append(key)
    
    #             if available_keys:
    #                 # Embaralhar para distribuir carga
    #                 return random.choice(available_keys)
    
    #             # Se todas estão em cooldown, usar a que sai primeiro
    #             next_available = min(self.api_keys,
    #                                  key=lambda k: self.key_status[k]["cooldown_until"])
    
    #             if current_time > self.key_status[next_available]["cooldown_until"]:
    #                 return next_available
    
    #             return None
    
    #         def _mark_key_error(self, api_key, error, cooldown_seconds=300):
    #             """
    #             Marca uma chave como problemática e define cooldown
    
    #             Args:
    #                 api_key: Chave com problema
    #                 error: Erro ocorrido
    #                 cooldown_seconds: Tempo de espera antes de tentar novamente
    #             """
    #             self.key_status[api_key]["available"] = False
    #             self.key_status[api_key]["last_error"] = str(error)
    #             self.key_status[api_key]["cooldown_until"] = time.time() + cooldown_seconds
    
    #             print(f"⚠️ Chave temporariamente indisponível: {api_key[:20]}... | Erro: {error}")
    
    #         def _mark_key_success(self, api_key):
    #             """
    #             Marca uma chave como funcionando corretamente
    
    #             Args:
    #                 api_key: Chave que funcionou
    #             """
    #             self.key_status[api_key]["available"] = True
    #             self.key_status[api_key]["last_error"] = None
    #             self.key_status[api_key]["cooldown_until"] = 0
    
    #         def classify_with_gemini_multi(self, image_input, max_attempts=None):
    #             """
    #             Classifica tipo corporal tentando múltiplas chaves Gemini.
    #             Aceita tanto um caminho de arquivo (str) quanto bytes da imagem.
    
    #             Args:
    #                 image_input: Caminho para a imagem (str) ou bytes da imagem (bytes).
    #                 max_attempts: Máximo de tentativas (None = tentar todas as chaves)
    
    #             Returns:
    #                 dict: Resultado da classificação com detalhes
    #             """
    #             print("🎯 Analisando com múltiplas chaves Gemini...")
    
    #             if max_attempts is None:
    #                 max_attempts = len(self.api_keys) * 2  # 2 tentativas por chave
    
    #             attempts = 0
    #             used_keys = []
    
    #             # Prompt otimizado para análise corporal
    #             prompt = """
    #             ANÁLISE DETALHADA DE TIPO CORPORAL:
    
    #             Observe CUIDADOSAMENTE esta pessoa e analise as proporções corporais:
    
    #             1. TRIANGULO (Pera):
    #                - Quadris/coxas CLARAMENTE mais largos que ombros
    #                - Cintura definida
    #                - Parte superior menor que inferior
    
    #             2. TRIANGULO_INVERTIDO (Maçã):
    #                - Ombros/busto CLARAMENTE mais largos que quadris
    #                - Torso mais volumoso
    #                - Parte superior maior que inferior
    
    #             3. OVAL:
    #                - Concentração de peso no abdômen/meio do corpo
    #                - Cintura POUCO ou NÃO definida
    #                - Formato arredondado no centro
    
    #             4. RETANGULO:
    #                - Ombros, cintura e quadris com larguras SIMILARES
    #                - Corpo reto/atlético
    #                - Pouca diferença entre medidas
    
    #             INSTRUÇÕES:
    #             - Compare VISUALMENTE as larguras
    #             - Ignore roupas largas, foque no formato corporal
    #             - Seja preciso na identificação
    #             - Considere a silhueta geral
    
    #             Responda APENAS: TRIANGULO, TRIANGULO_INVERTIDO, OVAL ou RETANGULO
    #             """
    
    #             while attempts < max_attempts:
    #                 attempts += 1
    
    #                 # Obter chave disponível
    #                 current_key = self._get_available_key()
    
    #                 if not current_key:
    #                     print("⏳ Todas as chaves estão em cooldown, aguardando...")
    #                     time.sleep(10)
    #                     continue
    
    #                 used_keys.append(current_key[:20] + "...")
    
    #                 try:
    #                     print(f"🔑 Tentativa {attempts} com chave: {current_key[:20]}...")
    
    #                     # Configurar Gemini com a chave atual
    #                     genai.configure(api_key=current_key)
    #                     model = genai.GenerativeModel('gemini-1.5-flash')
    
    #                     # --- Lógica de carregamento da imagem melhorada ---
    #                     if isinstance(image_input, str):  # Caminho de arquivo
    #                         if not os.path.exists(image_input):
    #                             raise ValueError(f"Arquivo não encontrado: {image_input}")
    #                         image = Image.open(image_input)
    #                     elif isinstance(image_input, bytes):  # Bytes da imagem
    #                         image = Image.open(io.BytesIO(image_input))
    #                     elif hasattr(image_input, 'save'):  # Objeto PIL.Image
    #                         image = image_input
    #                     else:
    #                         raise ValueError(
    #                             "Tipo de entrada inválido. Deve ser: caminho (str), bytes, ou PIL.Image")
    #                     # --- Fim da lógica de carregamento ---
    
    #                     # Gerar resposta
    #                     response = model.generate_content([prompt, image])
    #                     result = response.text.strip().upper()
    
    #                     # Normalizar resposta
    #                     result = self._normalize_response(result)
    
    #                     if result in self.valid_body_types:
    #                         print(f"✅ Sucesso com chave {current_key[:20]}... | Resultado: {result}")
    #                         self._mark_key_success(current_key)
    
    #                         return {
    #                             'result': result,
    #                             'key_used': current_key[:20] + "...",
    #                             'attempts': attempts,
    #                             'keys_tried': used_keys,
    #                             'success': True
    #                         }
    #                     else:
    #                         print(f"⚠️ Resposta inválida: {result}")
    
    #                 except Exception as e:
    #                     error_str = str(e).lower()
    
    #                     # Determinar tipo de erro e cooldown apropriado
    #                     if "quota" in error_str or "limit" in error_str:
    #                         cooldown = 3600  # 1 hora para quota exceeded
    #                         print(f"🚫 Quota excedida na chave {current_key[:20]}...")
    #                     elif "rate" in error_str:
    #                         cooldown = 60  # 1 minuto para rate limit
    #                         print(f"⏸️ Rate limit na chave {current_key[:20]}...")
    #                     else:
    #                         cooldown = 300  # 5 minutos para outros erros
    #                         print(f"❌ Erro na chave {current_key[:20]}...: {e}")
    
    #                     self._mark_key_error(current_key, e, cooldown)
    
    #                     # Pequena pausa antes da próxima tentativa
    #                     time.sleep(2)
    
    #             print(f"❌ Todas as tentativas falharam após {attempts} tentativas")
    #             return {
    #                 'result': None,
    #                 'key_used': None,
    #                 'attempts': attempts,
    #                 'keys_tried': used_keys,
    #                 'success': False
    #             }
    
    #         def _normalize_response(self, response):
    #             """
    #             Normaliza a resposta removendo acentos e caracteres especiais
    #             """
    #             # Remover caracteres especiais
    #             response = response.replace('Â', '').replace('Ã', '').replace('â', '').replace('ã', '')
    
    #             # Verificar se contém triângulo invertido primeiro (mais específico)
    #             if any(pattern in response for pattern in ['TRIANGULO_INVERTIDO', 'TRIÂNGULO_INVERTIDO']):
    #                 return 'TRIANGULO_INVERTIDO'
    
    #             # Depois verificar triângulo normal
    #             if any(pattern in response for pattern in ['TRIANGULO', 'TRIÂNGULO']) and 'INVERTIDO' not in response:
    #                 return 'TRIANGULO'
    
    #             # Verificar outros tipos
    #             if 'OVAL' in response:
    #                 return 'OVAL'
    #             if any(pattern in response for pattern in ['RETANGULO', 'RETÂNGULO']):
    #                 return 'RETANGULO'
    
    #             return response
    
    #         def classify_body_type(self, image_input):
    #             """
    #             Classifica tipo corporal usando múltiplas chaves Gemini.
    #             Aceita tanto um caminho de arquivo (str) quanto bytes da imagem.
    
    #             Args:
    #                 image_input: Caminho para a imagem (str) ou bytes da imagem (bytes).
    
    #             Returns:
    #                 dict: Resultado da classificação
    #             """
    #             print("=" * 60)
    #             print("🎯 CLASSIFICAÇÃO DE TIPO CORPORAL - MULTI GEMINI")
    #             print("=" * 60)
    
    #             # Classificar com múltiplas chaves
    #             result = self.classify_with_gemini_multi(image_input)
    
    #             if not result['success']:
    #                 return {
    #                     'result': None,
    #                     'method': 'Multi-Gemini',
    #                     'confidence': 0,
    #                     'details': 'Todas as chaves Gemini falharam',
    #                     'attempts': result['attempts'],
    #                     'keys_tried': result['keys_tried']
    #                 }
    
    #             # Formatar resultado
    #             formatted_result = self._format_body_type(result['result'])
    
    #             return {
    #                 'result': formatted_result,
    #                 'method': f'Multi-Gemini ({result["key_used"]})',
    #                 'confidence': 0.9,
    #                 'details': {
    #                     'raw_result': result['result'],
    #                     'attempts': result['attempts'],
    #                     'keys_tried': result['keys_tried']
    #                 }
    #             }
    
    #         def _format_body_type(self, body_type):
    #             """
    #             Formata o tipo corporal para exibição
    #             """
    #             body_type_names = {
    #                 'TRIANGULO': 'Triângulo',
    #                 'TRIANGULO_INVERTIDO': 'Triângulo Invertido',
    #                 'OVAL': 'Oval',
    #                 'RETANGULO': 'Retângulo (Atlético)'
    #             }
    
    #             return body_type_names.get(body_type, body_type)
    
    #         def get_keys_status(self):
    #             """
    #             Retorna o status atual de todas as chaves
    
    #             Returns:
    #                 dict: Status detalhado das chaves
    #             """
    #             current_time = time.time()
    #             status = {}
    
    #             for i, key in enumerate(self.api_keys):
    #                 key_info = self.key_status[key]
    #                 status[f"Chave {i + 1} ({key[:20]}...)"] = {
    #                     'disponível': key_info['available'] and current_time > key_info['cooldown_until'],
    #                     'último_erro': key_info['last_error'],
    #                     'cooldown_até': time.ctime(key_info['cooldown_until']) if key_info[
    #                                                                                   'cooldown_until'] > current_time else 'Nenhum'
    #                 }
    
    #             return status
    
    #     # Função simplificada para integração
    #     def classify_body_type_multi_gemini(image_input, gemini_api_keys=None):
    #         """
    #         Função simples para classificar tipo corporal com múltiplas chaves Gemini.
    #         Aceita tanto um caminho de arquivo (str) quanto bytes da imagem.
    
    #         Args:
    #             image_input: Caminho para a imagem (str) ou bytes da imagem (bytes).
    #             gemini_api_keys: Lista de chaves da API do Gemini
    
    #         Returns:
    #             str: Tipo corporal formatado
    #         """
    #         classifier = MultiGeminiBodyTypeClassifier(gemini_api_keys)
    #         result = classifier.classify_body_type(image_input)
    
    #         if result['result']:
    #             return result['result']
    #         else:
    #             return "Não foi possível classificar o tipo corporal"
    
    # # Exemplo de uso
    # # Suas múltiplas chaves Gemini
    # GEMINI_KEYS = [
    #     "AIzaSyBsuDaBYYHhNRLIob8U8Zbb1hKMWAuLASE",  # Chave principal
    #     "AIzaSyBujAPcUqckJ3vDceiXp2dcjoKSk5tB2jI",  # Chave backup 1
    #     "AIzaSyBdY1G2LdtQpsw1tAsuyNz5JED5T2gFt5w",  # Chave backup 2
    #     "AIzaSyApCqbHjrkpVMAMz07HxTKS4Hxas0SAONs"  # Chave backup 3
    # ]
    
    # # Caminho da imagem
    # image_path = caminho
    
    # try:
    #     print("🚀 CLASSIFICANDO COM MÚLTIPLAS CHAVES GEMINI")
    #     print("=" * 60)
    
    #     classifier = MultiGeminiBodyTypeClassifier(GEMINI_KEYS)
    #     result = classifier.classify_body_type(image_path)
    
    #     # Mostrar status das chaves
    #     print("\n📊 Status das chaves:")
    #     for key, status in classifier.get_keys_status().items():
    #         print(f"  {key}: {'✅' if status['disponível'] else '❌'}")
    
    #     print(f"\n✅ Resultado: {result['result']}")
    #     print(f"🔧 Método: {result['method']}")
    #     print(f"📊 Confiança: {result['confidence']}")
    #     print(f"🔄 Tentativas: {result['details']['attempts']}")
    #     medidas['Tipo de corpo'] = result['result']
    
    #     # Usando função simples
    #     # print("\n" + "=" * 60)
    #     # print("🎯 USANDO FUNÇÃO SIMPLES")
    #     # simple_result = classify_body_type_multi_gemini(image_path, GEMINI_KEYS)
    #     # print(f"✅ Resultado simples: {simple_result}")
    
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        import traceback
        traceback.print_exc()


    #cv2.imshow("Imagem de Entrada", imagem)

    # ================================= ROSTO  =================================
    h, w, _ = imagem.shape
    # NARIZ COMO CENTRO
    nariz = None
    if resultado.pose_landmarks:
        nariz = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        cx, cy = int(nariz.x * w), int(nariz.y * h)
    elif resultado_face.multi_face_landmarks:
        face_landmarks = resultado_face.multi_face_landmarks[0]
        nariz_face = face_landmarks.landmark[4]  # ponto da ponta do nariz
        cx, cy = int(nariz_face.x * w), int(nariz_face.y * h)
    else:
        print("Não foi possível localizar o nariz.")

    # DEFINIR ROI (zoom 3x ao redor do nariz)
    zoom_factor = 3
    roi_size = 150  # TAMANHO (pixels)
    x1 = max(cx - roi_size // 2, 0)
    y1 = max(cy - roi_size // 2, 0)
    x2 = min(cx + roi_size // 2, w)
    y2 = min(cy + roi_size // 2, h)

    roi = imagem[y1:y2, x1:x2]

    # AMPLIAR A ROI (zoom no rosto)
    if roi.size > 0:
        roi_ampliada = cv2.resize(roi, (roi_size * zoom_factor, roi_size * zoom_factor))

        # APLICAR FACE MESH NA ROI AMPLIADA
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
        ) as face_mesh:
            roi_rgb = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2RGB)
            resultado_face = face_mesh.process(roi_rgb)

            if resultado_face.multi_face_landmarks:
                face_landmarks = resultado_face.multi_face_landmarks[0]

                ponto_nariz = face_landmarks.landmark[4]  # ponta do nariz
                h_roi, w_roi, _ = roi_ampliada.shape
                x_nose, y_nose = int(ponto_nariz.x * w_roi), int(ponto_nariz.y * h_roi)

                offset = 8  # área mais precisa
                x1 = max(x_nose - offset, 0)
                y1 = max(y_nose - offset, 0)
                x2 = min(x_nose + offset, w_roi)
                y2 = min(y_nose + offset, h_roi)

                coordenadas_roi = (x1, y1, x2, y2)  # salva para usar no rosto saturado
                regiao_pele = roi_ampliada[y1:y2, x1:x2]

                if regiao_pele.size > 0:
                    # APLICA FILTRO HSV
                    regiao_hsv = cv2.cvtColor(regiao_pele, cv2.COLOR_BGR2HSV)
                    mask_pele = cv2.inRange(regiao_hsv, (0, 30, 60), (25, 150, 255))
                    regiao_filtrada = cv2.bitwise_and(regiao_pele, regiao_pele, mask=mask_pele)

                    # CALCULA MÉDIA DOS PIXELS DA PELE
                    tom_pele = cv2.mean(regiao_filtrada, mask=mask_pele)[:3]
                    medidas['tom_de_pele'] = np.array(tom_pele).astype(int)

                    # MOSTRA IMAGEM
                    debug_img = roi_ampliada.copy()
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                print("Landmarks faciais não detectados.")

    # =================================CABELO =================================
    if face_landmarks:
        try:
            h_roi, w_roi, _ = roi_ampliada.shape

            # ENCONTRA LINHA DO QUEIXO
            ponto_queixo = max(
                [(int(lm.x * w_roi), int(lm.y * h_roi)) for lm in face_landmarks.landmark[152:155]],
                key=lambda p: p[1]
            )

            # CONVERTE PARA HSV (Matiz, Saturação, Valor)
            hsv = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2HSV)

            # MASCARA LOIRO
            loiro_min = np.array([15, 40, 160])  # H, S, V
            loiro_max = np.array([45, 180, 255])
            mask_loiro = cv2.inRange(hsv, loiro_min, loiro_max)

            # MASCARA MORENO
            gray = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2GRAY)
            _, mask_escura = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

            # MASCARA DO ROSTO
            mask_rosto = np.zeros_like(gray)
            pontos_rosto = np.array([
                (int(lm.x * w_roi), int(lm.y * h_roi)) for lm in face_landmarks.landmark[:468]
            ])
            cv2.fillConvexPoly(mask_rosto, pontos_rosto, 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask_rosto = cv2.dilate(mask_rosto, kernel)

            # ISOLA A MASCARA DO CABELO
            mask_cabelo_total = cv2.bitwise_or(mask_loiro, mask_escura)
            mask_cabelo = cv2.subtract(mask_cabelo_total, mask_rosto)

            # RESTRINGE ÁREA ACIMA DO QUEIXO
            mask_regiao = np.zeros_like(gray)
            cv2.rectangle(mask_regiao, (0, 0), (w_roi, ponto_queixo[1] - 10), 255, -1)
            mask_cabelo = cv2.bitwise_and(mask_cabelo, mask_regiao)

            # ENCONTRA CONTORNOS
            contornos, _ = cv2.findContours(mask_cabelo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                maior_contorno = max(contornos, key=cv2.contourArea)
                mascara_final = np.zeros_like(gray)
                cv2.drawContours(mascara_final, [maior_contorno], -1, 255, -1)

                # MOSTRA IMAGEM
                cv2.drawContours(debug_img, [maior_contorno], -1, (0, 255, 0), 2)
                cv2.line(debug_img, (0, ponto_queixo[1] - 10), (w_roi, ponto_queixo[1] - 10), (0, 0, 255), 2)
                area_cabelo = cv2.countNonZero(mascara_final)
                limite_area_minima = 500  # Ajuste conforme seus testes (valor empírico)

                if area_cabelo < limite_area_minima:
                    medidas['pouco_cabelo'] = True
                    medidas['tom_de_cabelo'] = None
                else:
                    # EXTRAIR PIXELS DO CABELO
                    pixels_cabelo = cv2.bitwise_and(roi_ampliada, roi_ampliada, mask=mascara_final)
                    media_cabelo = cv2.mean(pixels_cabelo, mask=mascara_final)[:3]
                    medidas['tom_de_cabelo'] = np.array(media_cabelo).astype(int)
                    medidas['pouco_cabelo'] = False

            else:
                medidas['pouco_cabelo'] = True

        except Exception as e:
            print(f"Erro na análise de cabelo: {str(e)}")
            medidas['pouco_cabelo'] = True

    # ================================= OLHO =================================
    olho_esquerdo = [33, 133]
    olho_direito = [362, 263]

    for face_landmarks in resultado_face.multi_face_landmarks:
        h, w, _ = roi_ampliada.shape

        # COR DO OLHO ESQUERDO
        left_eye_coords = np.array([(int(face_landmarks.landmark[i].x * w),
                                     int(face_landmarks.landmark[i].y * h)) for i in olho_esquerdo])

        # REGIÃO EM VOLTA DO OLHO
        min_x = int(min(left_eye_coords[:, 0]))
        max_x = int(max(left_eye_coords[:, 0]))
        min_y = int(min(left_eye_coords[:, 1]))
        max_y = int(max(left_eye_coords[:, 1]))

        # EXTRAI A COR DO OLHO
        eye_region = roi_ampliada[min_y - 5: max_y + 5, min_x - 5: max_x + 5]

        # CALCULA A MÉDIA
        average_color = np.mean(eye_region, axis=(0, 1))
        medidas['tom_de_olho'] = np.round(average_color).astype(int)

        # ADICIONA NA IMAGEM
        cv2.rectangle(debug_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 1)
        #cv2.imshow("Rosto analisado", debug_img)

    # =================================CONTRASTE =================================
    def bgr_to_gray_scale_0_10(bgr):
        gray = int(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])
        escala = np.clip(round(gray / 255 * 10), 0, 10)
        return escala

    # OBTÉM ESCALA DE CINZA DOS TONS
    if 'tom_de_pele' in medidas and 'tom_de_cabelo' in medidas and medidas['tom_de_cabelo'] is not None:
        escala_pele = bgr_to_gray_scale_0_10(medidas['tom_de_pele'])
        escala_cabelo = bgr_to_gray_scale_0_10(medidas['tom_de_cabelo'])
        escala_olhos = bgr_to_gray_scale_0_10(medidas['tom_de_olho'])
    else:  # caso a pessoa não tenha cabelo ou tenha o cabelo com tom semelhante a pele
        escala_pele = bgr_to_gray_scale_0_10(medidas['tom_de_pele'])
        escala_cabelo = escala_pele
        escala_olhos = bgr_to_gray_scale_0_10(medidas['tom_de_olho'])

    # ENCONTRA TONS EXTREMOS
    tons = [escala_pele, escala_cabelo, escala_olhos]
    tom_min = min(tons)
    tom_max = max(tons)
    intervalo = tom_max - tom_min

    # CLASSIFICA O CONTRASTE
    if intervalo <= 3:
        if escala_pele <= 6:
            contraste = "Baixo contraste escuro"
        else:
            contraste = "Baixo contraste claro"
    elif intervalo <= 5:
        contraste = "Contraste médio"
    else:
        contraste = "Alto contraste"

    # ADICIONA NO DICIONÁRIO
    medidas["Tom de pele (escala 0-10)"] = escala_pele
    medidas["Tom de cabelo (escala 0-10)"] = escala_cabelo
    medidas["Tom dos olhos (escala 0-10)"] = escala_olhos
    medidas["Intervalo de contraste"] = intervalo
    medidas["Classificação"] = contraste

    # ================================= ROSTO COM MAIOR VIBRAÇÃO =================================
    def vibrance_contraste_suave(roi_ampliada):
        lab = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        img_clahe = cv2.merge((l_clahe, a, b))
        img_bgr_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)

        # conversão para HSV
        hsv = cv2.cvtColor(img_bgr_clahe, cv2.COLOR_BGR2HSV).astype("float32")
        h, s, v = cv2.split(hsv)

        # vibrance: aumenta onde a saturação é baixa
        vibrance_mask = s < 150  # onde a saturação é média ou baixa
        s[vibrance_mask] *= 1.25  # aumento seletivo
        s = np.clip(s, 0, 255)

        hsv_vibrant = cv2.merge([h, s, v])
        result_bgr = cv2.cvtColor(hsv_vibrant.astype("uint8"), cv2.COLOR_HSV2BGR)

        return result_bgr

    # CARREGA A IMAGEM
    img = cv2.imread("/mnt/data/45c92eeb-79d9-4194-90d0-83d1a410258b.png")
    imagem_realcada = vibrance_contraste_suave(roi_ampliada)

    # APLICA FACE MESH A NA IMAGEM REALÇADA
    coordenadas_roi = (x1, y1, x2, y2)
    regiao_pele = imagem_realcada[y1:y2, x1:x2]

    if regiao_pele.size > 0:
        # APLICA FILTRO HSV
        regiao_hsv = cv2.cvtColor(regiao_pele, cv2.COLOR_BGR2HSV)
        mask_pele = cv2.inRange(regiao_hsv, (0, 30, 60), (25, 150, 255))
        regiao_filtrada = cv2.bitwise_and(regiao_pele, regiao_pele, mask=mask_pele)

        # CALCULA MÉDIA DO TOM DA PELE
        tom_pele = cv2.mean(regiao_filtrada, mask=mask_pele)[:3]
        medidas['cor_saturada'] = np.array(tom_pele).astype(int)

        # MOSTRA A IMAGEM REALÇADA
        debug_img = imagem_realcada.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.imshow("Rosto novo", debug_img)

    # ================================= COMPARANDO RESULTADOS =================================
    # SUBTONS DE REFERÊNCIA
    subtons_bgr = {
        "baixo contraste escuro": {
            "Frio": [81, 113, 219],
            "Neutro": [80, 117, 214],
            "Quente": [66, 112, 207],
            "Oliva": [66, 113, 185]
        },
        "baixo contraste claro": {
            "Frio": [175, 188, 233],
            "Neutro": [180, 196, 231],
            "Quente": [170, 198, 230],
            "Oliva": [180, 205, 235]
        },
        "medio contraste": {
            "Frio": [138, 169, 255],
            "Neutro": [135, 169, 254],
            "Quente": [114, 158, 246],
            "Oliva": [120, 169, 240]
        }
    }

    # CONVERTE BGR PARA LAB (luminosidade e componente de cores)
    def bgr_para_lab(bgr):
        pix = np.uint8([[bgr]])
        lab = cv2.cvtColor(pix, cv2.COLOR_BGR2LAB)
        return lab[0, 0]

    # DISTÂNCIA EUCLIDIANA ENTRE 2 TONS LAB
    def distancia_lab(lab1, lab2):
        return np.linalg.norm(np.array(lab1, float) - np.array(lab2, float))

    # CLASSIFICAÇÃO DO SUBTOM BASEADO NO BGR DE ENTRADA
    def classificar_subtom(bgr_input):
        if medidas["Classificação"] == "Baixo contraste escuro":
            subtons_select = subtons_bgr["baixo contraste escuro"]
        if medidas["Classificação"] == "Baixo contraste claro":
            subtons_select = subtons_bgr["baixo contraste claro"]
        else:
            subtons_select = subtons_bgr["medio contraste"]

        # CONVERTE OS SUBTONS PARA LAB
        subtons_lab = {nome: bgr_para_lab(bgr) for nome, bgr in subtons_select.items()}

        # CONVERTE O TOM DE ENTRADA PARA LAB
        lab_input = bgr_para_lab(bgr_input)

        # CALCULA AS DISTÂNCIAS E ENCONTRA O SUBTOM MAIS PRÓXIMO
        distancias = {
            nome: distancia_lab(lab_input, lab_base)
            for nome, lab_base in subtons_lab.items()
        }
        subtom_proximo = min(distancias, key=distancias.get)

        return subtom_proximo, distancias

    # APLICA NA COR SATURADA E ADICIONA NO DICIONÁRIO
    subtom, dist = classificar_subtom(medidas['cor_saturada'])
    medidas['Subtom'] = subtom
    for k, v in dist.items():
        v = int(v)
        dist[k] = v
    medidas['Distâncias'] = dist

    # =========== FORMATO DO ROSTO ===========
    def calcular_distancia(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    if resultado_face.multi_face_landmarks:
        for rosto in resultado_face.multi_face_landmarks:
            h, w, _ = imagem.shape
            pontos = [(int(p.x * w), int(p.y * h)) for p in rosto.landmark]
            # pontos principais
            topo_testa = pontos[10]
            queixo = pontos[152]
            mandibula_esq = pontos[234]
            mandibula_dir = pontos[454]
            lateral_testa_esq = pontos[127]
            lateral_testa_dir = pontos[356]
            # medidas principais
            altura_rosto = calcular_distancia(topo_testa, queixo)
            largura_mandibula = calcular_distancia(mandibula_esq, mandibula_dir)
            largura_testa = calcular_distancia(lateral_testa_esq, lateral_testa_dir)
            # classificar
            prop = altura_rosto / largura_mandibula

            if largura_testa > largura_mandibula and prop > 1.3:
                formato = 'Coração'
            elif abs(largura_testa - largura_mandibula) < 15 and prop > 1.3:
                formato = 'Oval'
            elif abs(largura_testa - largura_mandibula) < 15 and prop < 1.2:
                formato = 'Redondo'
            else:
                formato = 'Quadrado'
            medidas['Formato do rosto'] = formato

    else:
        print('Nenhum rosto detectado')

    # ============ INTENSIDADE ==============
    def intensidade_saturacao(bgr):
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
        return hsv[0, 0, 1]  # Saturação varia de 0 a 255

    s_p = intensidade_saturacao(medidas['tom_de_pele'])
    s_o = intensidade_saturacao(medidas['tom_de_olho'])
    s_c = intensidade_saturacao(medidas['tom_de_cabelo']) if medidas['tom_de_cabelo'] is not None else s_p  # fallback

    # Peso maior na pele
    intensidade_media = (0.5 * s_p + 0.3 * s_o + 0.2 * s_c)

    if intensidade_media >= 140:
        intensidade = "Alta"
    elif intensidade_media >= 90:
        intensidade = "Média"
    else:
        intensidade = "Baixa"

    medidas['Intensidade'] = intensidade
    medidas['Valor Saturação'] = int(intensidade_media)

    # =============== profundidade ==============

    def obter_luminosidade(bgr):
        lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)
        return lab[0, 0, 0]  # Luminância (0 a 255)

    l_p = obter_luminosidade(medidas['tom_de_pele'])
    l_o = obter_luminosidade(medidas['tom_de_olho'])
    l_c = obter_luminosidade(medidas['tom_de_cabelo']) if medidas['tom_de_cabelo'] is not None else l_p

    # Peso maior na pele e cabelo
    luminosidade = (0.5 * l_p + 0.3 * l_c + 0.2 * l_o)

    if luminosidade > 160:
        profundidade = "Claro"
    else:
        profundidade = "Escuro"

    medidas["Profundidade"] = profundidade
    medidas["Luminosidade Média"] = int(luminosidade)

    return medidas, resultado


def visualizar_resultados(imagem, resultado, tom_de_pele=None, pouco_cabelo=None, tom_de_cabelo=None, tom_de_olho=None):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # PROCESSA A IMAGEM ORIGINAL
    imagem_landmarks = imagem.copy()

    # LANDMARKS CORPORAIS
    if resultado.pose_landmarks:
        mp_drawing.draw_landmarks(
            imagem_landmarks,
            resultado.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    # CRIA PAINEL DE RESULTADOS
    painel_resultados = np.full((imagem.shape[0], 300, 3), 240, dtype=np.uint8)

    # CABEÇALHO
    cv2.putText(painel_resultados, "RESULTADOS", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    y_atual = 60  # controle da posição vertical

    # ========== TOM DE PELE ==========
    cv2.putText(painel_resultados, "Tom de pele:", (20, y_atual + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(painel_resultados, (20, y_atual + 20), (120, y_atual + 100),
                  tuple([int(c) for c in tom_de_pele]), -1)
    cv2.putText(painel_resultados, f"RGB: {list(tom_de_pele)}", (20, y_atual + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    y_atual += 140

    # ========== TOM DE CABELO ==========
    cv2.putText(painel_resultados, "Tom de cabelo:", (20, y_atual + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if not pouco_cabelo and tom_de_cabelo is not None:
        cv2.rectangle(painel_resultados, (20, y_atual + 30), (120, y_atual + 120),
                      tuple([int(c) for c in tom_de_cabelo]), -1)
        cv2.putText(painel_resultados, f"RGB: {list(tom_de_cabelo)}", (20, y_atual + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        cv2.putText(painel_resultados, "Nao detectado", (20, y_atual + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    y_atual += 160

    # ========== TOM DE OLHO ==========
    cv2.putText(painel_resultados, "Tom de olho:", (20, y_atual + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(painel_resultados, (20, y_atual + 20), (120, y_atual + 100),
                  tuple([int(c) for c in tom_de_olho]), -1)
    cv2.putText(painel_resultados, f"RGB: {list(tom_de_olho)}", (20, y_atual + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # COMBINA AS IMAGEMS HORIZONTALMENTE
    imagem_final = np.hstack((imagem_landmarks, painel_resultados))

    # EXIBIÇÃO
    cv2.namedWindow("Analise Corporal", cv2.WINDOW_NORMAL)
    #cv2.imshow("Analise Corporal", imagem_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
