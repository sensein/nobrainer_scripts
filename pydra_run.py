import pydra


#drp=["None", "bernoulli", "concrete"]

cmd=["python3", "run_kwyk_mirror_trainbatch.py"]
kwyk_image="/om2/user/hodaraja/containers/tf2_nobrainer_0.sif"

kwyk_input_spec = pydra.specs.SpecInfo(
	name="Input",
	fields=[
		("dim", int, {"mandatory":True, "position": 1, "argstr": "", "help_string": "input volume dimension"}),
		("dropout", str, {"mandatory":True, "position": 2, "argstr": "", "help_string": "model dropout type"})
	],
	bases=(pydra.specs.SingularitySpec),
)

kwyk_task = pydra.SingularityTask(
	name="kwyk", executable=cmd,
	image=kwyk_image,
	container_xargs=["--nv"],
	bindings=[("/om/user/satra/kwyk/tfrecords/", "/om2/user/hodaraja")],
	cache_dir=tmpdir,
	dim=128,
	dropout="None",
	input_spec= kwyk_input_spec
)

with pydra.Submitter(plugin="cf") as sub:
	sub(kwyk_task)












cmd = ["-m", "bvwn_multi_prior", "-n", "2", "--save-variance", "--save-entropy"]

my_input_spec = SpecInfo(
    name="Input",
    fields=[
        ("T1", File, {"mandatory": True, "position": 1, "argstr": "", "help_string": "T1 file"}),
        ("out_prefix", str, {"position": 2, "argstr": "", "help_string": "output prfix"})
    ],
    bases=(SingularitySpec,),
)

singu = SingularityTask(
    name="singu", executable=cmd,
    image="kwyk_latest-gpu.sif", container_xargs=["-W", "/data", "--nv"],
    bindings=[("your_path", "/data", "ro")],
    cache_dir=tmpdir,
    T1="/Users/dorota/tmp/withoutLesion/T1.nii",
    out_prefix = "withoutLesion/output",
    input_spec=my_input_spec
)