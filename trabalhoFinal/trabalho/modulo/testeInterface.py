from gradio_client import Client, handle_file, file

# client = Client("yisol/IDM-VTON")
# print(client.view_api())
#
# pessoa = r"C:\Users\HOME\PycharmProjects\trabalhoFinal\trabalho\data\imagens_testes\download.jpg"
roupa_1 = r"C:\Users\HOME\PycharmProjects\trabalhoFinal\trabalho\data\imagens_roupas\photo_5028271119014146085_x.jpg"
roupa_2 = r"C:\Users\HOME\Downloads\20230914115132_2540997460_GZ.png"
#
# result = client.predict(
# 		dict={"background":file(pessoa),"layers":[],"composite":null},
# 		garm_img=file(roupa_1),
# 		garment_des="Hello!!",
# 		is_checked=True,
# 		is_checked_crop=False,
# 		denoise_steps=30,
# 		seed=42,
# 		api_name="/tryon")
#
# with open("resultado.jpg", "wb") as f:
#     f.write(result)

from gradio_client import Client
import shutil

client = Client("yisol/IDM-VTON")


# Caminhos para as imagens
model_img = r"C:\Users\HOME\PycharmProjects\trabalhoFinal\trabalho\data\imagens_testes\download.jpg"      # modelo base (foto da pessoa)
camisa_img = "/mnt/data/camisa.png"     # segunda roupa a ser sobreposta
blusa_img = r"C:\Users\HOME\PycharmProjects\trabalhoFinal\trabalho\data\imagens_roupas\photo_5028271119014146085_x.jpg"       # primeira roupa

# Enviando para a API com duas camadas (roupas) sobrepostas
result = client.predict(
    dict={
        "background": handle_file(model_img),
        "layers": [], #[handle_file(camisa_img)],  #Adiciona a camisa como camada extra
        "composite": None
    },
    garm_img=handle_file(blusa_img),           # blusa será a primeira peça colocada
    garment_des="Blusa branca elegante.",
    is_checked=True,
    is_checked_crop=False,
    denoise_steps=30,
    seed=42,
    api_name="/tryon"
)

# Salvar resultado
output_path, masked_path = result
shutil.copy(output_path, "resultado_com_duas_roupas.png")
print("Resultado salvo como 'resultado_com_duas_roupas.png'")

