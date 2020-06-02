from raytraverse import Scene, Sampler

scene = Scene('test.oct', 'planesm.rad', 'test_run', wea='geneva.epw',
              reload=True)
sampler = Sampler(scene, ipres=1, ptres=2, minrate=.01)
# sampler.mkpmap('glz sglz')
# sampler.skypmap = True
sampler.run(rcopts='-ab 3 -ad 8000 -as 4000 -lw 1e-4')
