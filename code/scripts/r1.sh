python hyperoptim.py -c "python compute_sigma.py -i /local2/pratikac/results/allcnns/\(Sep_08_02_44_57\)_opt_\{\"b\":128,\"dataset\":\"cifar10\",\"m\":\"allcnns\",\"s\":42\}/allcnns_50.pz" -p '{"b":[128,1024]}' -j 1

python hyperoptim.py -c "python compute_sigma.py -i /local2/pratikac/results/allcnns/\(Sep_08_02_45_19\)_opt_\{\"b\":128,\"dataset\":\"cifar100\",\"m\":\"allcnns\",\"s\":42\}/allcnns_50.pz" -p '{"b":[128,1024]}' -j 1
