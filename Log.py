def registrar(arquivo, conteudo):
    f = open(arquivo, "a")
    f.write(conteudo)
    f.close()
