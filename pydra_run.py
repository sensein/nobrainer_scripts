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

