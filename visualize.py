import matplotlib.pyplot as plt 


H, W = (50, 50)
R = [5, 10]
UE = [x/2 for x in R]
cell_W, cell_H = (10, 10)
targets = []
for h in range(int(abs(H/cell_H))):
    for w in range(int(abs(W/cell_W))):
        targets.append((w*cell_W + cell_W/2, h*cell_H + cell_H/2))
xtarget = [x[0] for x in targets]
ytarget = [x[1] for x in targets]

# harmony = [[36.52235284213187, 5.158923571575547], [5.803977184212332, 16.579737439122482], [19.76909132203916, -1], [16.362401717114682, 42.94448301719304], [39.27175114074775, 10.992642627364102], [16.74205055331246, 38.35399254373819], [27.312578446735305, 13.395017147960669], [43.46952776424973, 30.504267411728666], [-1, 36.5782468817317], [19.529226257956356, 13.674552516931001], [39.30332867498081, 18.56383898679198], [3.3012767353284587, 4.322705960180589], [10.84872319141316, 7.147392967931629], [24.93388616312154, 40.157932114494756], [33.76329870791216, 41.99536186646283], [6.990908167067408, -1], [37.14331736420686, 34.45115519008454], [19.333452467268657, -1], [8.07785057792524, 12.428832873457463], [42.564132363103894, 24.558477993553797], [26.89983254707116, 20.237557138374008], [13.859190969161306, 31.647922005541822], [40.26192064408836, -1], [21.40285292713656, 21.946540792085653], [-1, 33.87291542486836]]
# types = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]




def draw(harmony, types, savepath):
   
    xused = []
    yused = []
    for x in harmony:
        if x[0] == -1 or x[1] == -1:
            continue
        xused.append(x[0])
        yused.append(x[1])

    fig, ax1 = plt.subplots(1)

    ax1.scatter(xtarget, ytarget, marker=".")
    ax1.scatter(xused, yused, marker="*")
    for s in range(len(xused)):
        ax1.add_patch(plt.Circle((xused[s], yused[s]), R[types[s]], color='r', alpha=0.5, fill=False))
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.plot()
    ax1.grid()
    plt.savefig(savepath)
    # plt.show()
