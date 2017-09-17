python compute_sigma.py -i '/local2/pratikac/results/allcnns/(Sep_08_02_45_19)_opt_{"b":128,"dataset":"cifar100","m":"allcnns","s":42}/allcnns_00.pz' -g 0 &
python compute_sigma.py -i '/local2/pratikac/results/allcnns/(Sep_08_02_45_19)_opt_{"b":128,"dataset":"cifar100","m":"allcnns","s":42}/allcnns_30.pz' -g 1 &
python compute_sigma.py -i '/local2/pratikac/results/allcnns/(Sep_08_02_45_19)_opt_{"b":128,"dataset":"cifar100","m":"allcnns","s":42}/allcnns_50.pz' -g 2

# python compute_sigma.py -i '/local2/pratikac/results/allcnns/(Sep_08_02_45_19)_opt_{"b":128,"dataset":"cifar100","m":"allcnns","s":42}/allcnns_00.pz' --stats
# python compute_sigma.py -i '/local2/pratikac/results/allcnns/(Sep_08_02_45_19)_opt_{"b":128,"dataset":"cifar100","m":"allcnns","s":42}/allcnns_30.pz' --stats
# python compute_sigma.py -i '/local2/pratikac/results/allcnns/(Sep_08_02_45_19)_opt_{"b":128,"dataset":"cifar100","m":"allcnns","s":42}/allcnns_50.pz' --stats
