session = 'sub-830794/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys'
# session = 'sub-830795/sub-830795_ses-ecephys-830795-2026-02-23-15-03-59_ecephys'

for trial in 0 1; do
	for probe in A B C D E F; do
		for phase in 0 1; do
#			COMMAND="python optimize_waven_parameters.py /user/raabe14/u19361/workspace-allen/results/zebra/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys/Probe${probe}/trial_${trial}/phase_${phase}/lib_ab4a70858e/ -o ../../../results/allen_open_scope/rf/waven/zebra/optimized/sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys/optimized__sub-830794_ses-ecephys-830794-2026-01-26-12-02-05_ecephys__Probe${probe}__trial_${trial}__phase_${phase}.h5"
			COMMAND="python optimize_waven_parameters.py /user/raabe14/u19361/workspace-allen/results/zebra/${session}.nwb/Probe${probe}/trial_${trial}/phase_${phase}/lib_ab4a70858e/ -o ../../../results/allen_open_scope/rf/waven/zebra/optimized/${session}/optimized__${session}__Probe${probe}__trial_${trial}__phase_${phase}.h5"
			echo $COMMAND 
			$COMMAND &
		done
	done
done
