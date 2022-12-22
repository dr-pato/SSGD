import sys
import os
import subprocess
from glob import glob
import re
import shutil
import pandas as pd
from pyannote.metrics.segmentation import Annotation, Segment


MD_EVAL_PATH = '/home/gmorrone/code/EEND/tools/kaldi/tools/sctk/bin/md-eval.pl'


def der_eval(rttm_ref_list, rttm_sys_list, collar=0.25, ignovr=False):
    # RTTM result list init
    id_list = []
    eval_time_list = []
    scored_time_list = []
    scored_speaker_time_list = []
    missed_speaker_time_list = []
    falarm_speaker_time_list = []
    speaker_error_time_list = []

    for ref_rttm, sys_rttm in zip(rttm_ref_list, rttm_sys_list):
        # Compute DER
        command_args = [MD_EVAL_PATH, '-c', str(collar), '-r', ref_rttm, '-s', sys_rttm]
        if ignovr:
            command_args.append('-1')
        try:
            der_output = subprocess.check_output(command_args, stderr=subprocess.DEVNULL).decode()
        except subprocess.CalledProcessError:
            print('Error in executing DER computation. Closing...')
            exit(1)
        
        id_list.append(os.path.basename(ref_rttm.replace('.rttm', '')))
        # Extract results from output
        eval_time = re.findall('EVAL TIME.*$', der_output, re.MULTILINE)[0]
        eval_time = float(eval_time.split()[3])
        eval_time_list.append(eval_time)
        
        scored_time = re.findall('SCORED TIME.*$', der_output, re.MULTILINE)[0]
        scored_time = float(scored_time.split()[3])
        scored_time_list.append(scored_time)
        
        scored_speaker_time = re.findall('SCORED SPEAKER TIME.*$', der_output, re.MULTILINE)[0]
        scored_speaker_time = float(scored_speaker_time.split()[4])
        scored_speaker_time_list.append(scored_speaker_time)
        
        missed_speaker_time = re.findall('MISSED SPEAKER TIME.*$', der_output, re.MULTILINE)[0]
        missed_speaker_time = float(missed_speaker_time.split()[4])
        missed_speaker_time_list.append(missed_speaker_time)
        
        falarm_speaker_time = re.findall('FALARM SPEAKER TIME.*$', der_output, re.MULTILINE)[0]
        falarm_speaker_time = float(falarm_speaker_time.split()[4])
        falarm_speaker_time_list.append(falarm_speaker_time)
        
        speaker_error_time = re.findall('SPEAKER ERROR TIME.*$', der_output, re.MULTILINE)[0]
        speaker_error_time = float(speaker_error_time.split()[4])
        speaker_error_time_list.append(speaker_error_time)

    # Aggregate the results
    results = pd.DataFrame(zip(id_list, eval_time_list, scored_time_list, scored_speaker_time_list, missed_speaker_time_list, falarm_speaker_time_list, speaker_error_time_list),
                           columns=['ID', 'EVAL TIME', 'SCORED TIME', 'SCORED SPEAKER TIME', 'MISSED', 'FALSE ALARM', 'SPEAKER ERROR'])
    results['MISSED (%)'] = results['MISSED'] / results['SCORED SPEAKER TIME'] * 100
    results['FALSE ALARM (%)'] = results['FALSE ALARM'] / results['SCORED SPEAKER TIME'] * 100
    results['SPEAKER ERROR (%)'] = results['SPEAKER ERROR'] / results['SCORED SPEAKER TIME'] * 100
    results['DER (%)'] = (results['MISSED'] + results['FALSE ALARM'] + results['SPEAKER ERROR']) / results['SCORED SPEAKER TIME'] * 100

    # Compute overall metrics
    overall_der = (results['MISSED'].sum() + results['FALSE ALARM'].sum() + results['SPEAKER ERROR'].sum()) / results['SCORED SPEAKER TIME'].sum() * 100
    overall_miss = results['MISSED'].sum() / results['SCORED SPEAKER TIME'].sum() * 100
    overall_fa = results['FALSE ALARM'].sum() / results['SCORED SPEAKER TIME'].sum() * 100
    overall_spkerr = results['SPEAKER ERROR'].sum() / results['SCORED SPEAKER TIME'].sum() * 100

    return results.sort_values('ID'), overall_der, overall_miss, overall_fa, overall_spkerr


def save_list(lines, out_path):
    with open(out_path, 'w') as f:
        for l in lines:
            f.write('%s\n' % l)


def der_eval_overall(rttm_ref_list, rttm_sys_list, collar=0.25, ignovr=False):
    # Create file lists
    tmp_dir = os.path.join('/tmp', str(os.getpid()) + '_dereval')
    os.makedirs(tmp_dir, exist_ok=True)
    ref_list_path = os.path.join(tmp_dir, 'ref_list.txt')
    sys_list_path = os.path.join(tmp_dir, 'sys_list.txt')
    save_list(rttm_ref_list, ref_list_path)
    save_list(rttm_sys_list, sys_list_path)
    
    # Compute DER
    command_args = [MD_EVAL_PATH, '-c', str(collar), '-R', ref_list_path, '-S', sys_list_path]
    if ignovr:
        command_args.append('-1')
    try:
        der_output = subprocess.check_output(command_args, stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        print('Error in executing DER computation. Closing...')
        exit(1)

    # Remove temp files
    shutil.rmtree(tmp_dir)

    eval_time = re.findall('EVAL TIME.*$', der_output, re.MULTILINE)[0]
    eval_time = float(eval_time.split()[3])
    
    scored_time = re.findall('SCORED TIME.*$', der_output, re.MULTILINE)[0]
    scored_time = float(scored_time.split()[3])
    
    scored_speaker_time = re.findall('SCORED SPEAKER TIME.*$', der_output, re.MULTILINE)[0]
    scored_speaker_time = float(scored_speaker_time.split()[4])
    
    missed_speaker_time = re.findall('MISSED SPEAKER TIME.*$', der_output, re.MULTILINE)[0]
    missed_speaker_time = float(missed_speaker_time.split()[4])
    
    falarm_speaker_time = re.findall('FALARM SPEAKER TIME.*$', der_output, re.MULTILINE)[0]
    falarm_speaker_time = float(falarm_speaker_time.split()[4])
    
    speaker_error_time = re.findall('SPEAKER ERROR TIME.*$', der_output, re.MULTILINE)[0]
    speaker_error_time = float(speaker_error_time.split()[4])
    

    overall_der = (missed_speaker_time + falarm_speaker_time + speaker_error_time) / scored_speaker_time * 100
    overall_miss = missed_speaker_time / scored_speaker_time * 100
    overall_fa = falarm_speaker_time / scored_speaker_time * 100
    overall_spkerr = speaker_error_time / scored_speaker_time * 100
    
    return overall_der, overall_miss, overall_fa, overall_spkerr
   


if __name__ == '__main__':
    model_name = sys.argv[1]
    rttm_ref_dir = sys.argv[2]
    rttm_sys_dir = sys.argv[3]
    out_res_dir = "" if sys.argv[4] == 'None' else sys.argv[4]


    rttm_ref_list = sorted(glob(os.path.join(rttm_ref_dir, '*.rttm')))
    rttm_sys_list = sorted(glob(os.path.join(rttm_sys_dir, '*.rttm')))

    print('Compute DER...')
    if out_res_dir:
        # Create results output directory
        os.makedirs(out_res_dir, exist_ok=True)
    
        # Save results in CSV files
        res_ignovr_collar250, der_ignovr_collar250, miss_ignovr_collar250, fa_ignovr_collar250, spkerr_ignovr_collar250 = der_eval(rttm_ref_list, rttm_sys_list, collar=0.25, ignovr=True)
        out_res_file = os.path.join(out_res_dir, model_name + '_ignovr_collar0.25.csv')
        res_ignovr_collar250.to_csv(out_res_file, index=False)

        res_collar250, der_collar250, miss_collar250, fa_collar250, spkerr_collar250 = der_eval(rttm_ref_list, rttm_sys_list, collar=0.25)
        out_res_file = os.path.join(out_res_dir, model_name + '_collar0.25.csv')
        res_collar250.to_csv(out_res_file, index=False)
        
        res_collar0, der_collar0, miss_collar0, fa_collar0, spkerr_collar0 = der_eval(rttm_ref_list, rttm_sys_list, collar=0)
        out_res_file = os.path.join(out_res_dir, model_name + '_collar0.csv')
        res_collar0.to_csv(out_res_file, index=False)

        # Print results
        print('Forgiving | DER: {:.2f} ({:.2f}) - MISS: {:.2f} ({:.2f}) - FA: {:.2f} ({:.2f}) - SPEAKER ERR: {:.2f} ({:.2f})'\
              .format(der_ignovr_collar250, res_ignovr_collar250['DER (%)'].std(), miss_ignovr_collar250, res_ignovr_collar250['MISSED (%)'].std(),
                      fa_ignovr_collar250, res_ignovr_collar250['FALSE ALARM (%)'].std(), spkerr_ignovr_collar250, res_ignovr_collar250['SPEAKER ERROR (%)'].std()))
        
        print('Fair      | DER: {:.2f} ({:.2f}) - MISS: {:.2f} ({:.2f}) - FA: {:.2f} ({:.2f}) - SPEAKER ERR: {:.2f} ({:.2f})'\
              .format(der_collar250, res_collar250['DER (%)'].std(), miss_collar250, res_collar250['MISSED (%)'].std(),
                      fa_collar250, res_collar250['FALSE ALARM (%)'].std(), spkerr_collar250, res_collar250['SPEAKER ERROR (%)'].std()))

        print('Full      | DER: {:.2f} ({:.2f}) - MISS: {:.2f} ({:.2f}) - FA: {:.2f} ({:.2f}) - SPEAKER ERR: {:.2f} ({:.2f})'\
              .format(der_collar0, res_collar0['DER (%)'].std(), miss_collar0, res_collar0['MISSED (%)'].std(),
                      fa_collar0, res_collar0['FALSE ALARM (%)'].std(), spkerr_collar0, res_collar0['SPEAKER ERROR (%)'].std()))
    else:
        der_ignovr_collar250, miss_ignovr_collar250, fa_ignovr_collar250, spkerr_ignovr_collar250 = der_eval_overall(rttm_ref_list, rttm_sys_list, collar=0.25, ignovr=True)
        der_collar250, miss_collar250, fa_collar250, spkerr_collar250 = der_eval_overall(rttm_ref_list, rttm_sys_list, collar=0.25)
        der_collar0, miss_collar0, fa_collar0, spkerr_collar0 = der_eval_overall(rttm_ref_list, rttm_sys_list, collar=0)
        
        # Print results
        print('Forgiving | DER: {:.2f} - MISS: {:.2f} - FA: {:.2f} - SPEAKER ERR: {:.2f}'\
              .format(der_ignovr_collar250, miss_ignovr_collar250, fa_ignovr_collar250, spkerr_ignovr_collar250))
        
        print('Fair      | DER: {:.2f} - MISS: {:.2f} - FA: {:.2f} - SPEAKER ERR: {:.2f}'\
              .format(der_collar250, miss_collar250, fa_collar250, spkerr_collar250))

        print('Full      | DER: {:.2f} - MISS: {:.2f} - FA: {:.2f} - SPEAKER ERR: {:.2f}'\
              .format(der_collar0, miss_collar0, fa_collar0, spkerr_collar0))


def convert2pyannote(refs, hyps):
    ref_ann = Annotation()
    for spk, start, stop in refs:
        ref_ann[Segment(start, stop)] = spk

    hyp_ann = Annotation()
    for spk, start, stop in hyps:
        hyp_ann[Segment(start, stop)] = spk

    return ref_ann, hyp_ann