from ctr.reconstruction.error import transform

corresp_coors = [[132,129],[328,148]]

for i in range(5):
    base = transform(corresp_coors)

    print(base)