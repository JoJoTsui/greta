localrules: prior_tfm, prior_tfp, prior_cre


rule prior_tfm:
    threads: 1
    # singularity: 'workflow/envs/gretabench.sif'
    conda: 'gretabench'
    input:
        grn=lambda wildcards: rules.grn_run.output.out.format(**wildcards),
        db='dbs/hg38/tfm/{db}/{db}.tsv',
    output:
        out='anl/metrics/prior/tfm/{db}/{dat}.{case}/{pre}.{p2g}.{tfb}.{mdl}.scores.csv'
    shell:
        """
        python workflow/scripts/anl/metrics/prior/tfm.py \
        -a {input.grn} \
        -b {input.db} \
        -f {output.out}
        """


rule prior_tfp:
    threads: 1
    # singularity: 'workflow/envs/gretabench.sif'
    conda: 'gretabench'
    input:
        grn=lambda wildcards: rules.grn_run.output.out.format(**wildcards),
        db='dbs/hg38/tfp/{db}/{db}.tsv',
    output:
        out='anl/metrics/prior/tfp/{db}/{dat}.{case}/{pre}.{p2g}.{tfb}.{mdl}.scores.csv'
    params:
        thr_p=0.01,
    shell:
        """
        python workflow/scripts/anl/metrics/prior/tfp.py \
        -a {input.grn} \
        -b {input.db} \
        -p {params.thr_p} \
        -f {output.out}
        """


rule prior_tfb:
    threads: 1
    # singularity: 'workflow/envs/gretabench.sif'
    conda: 'gretabench'
    input:
        grn=lambda wildcards: rules.grn_run.output.out.format(**wildcards),
        db='dbs/hg38/tfb/{db}/{db}.bed',
    output:
        out='anl/metrics/prior/tfb/{db}/{dat}.{case}/{pre}.{p2g}.{tfb}.{mdl}.scores.csv'
    params:
        grp='source',
    shell:
        """
        python workflow/scripts/anl/metrics/prior/gnm.py \
        -a {input.grn} \
        -b {input.db} \
        -d {params.grp} \
        -f {output}
        """


rule prior_cre:
    threads: 1
    # singularity: 'workflow/envs/gretabench.sif'
    conda: 'gretabench'
    input:
        grn=lambda wildcards: rules.grn_run.output.out.format(**wildcards),
        db='dbs/hg38/cre/{db}/{db}.bed',
    output:
        out='anl/metrics/prior/cre/{db}/{dat}.{case}/{pre}.{p2g}.{tfb}.{mdl}.scores.csv'
    shell:
        """
        python workflow/scripts/anl/metrics/prior/gnm.py \
        -a {input.grn} \
        -b {input.db} \
        -f {output}
        """


# ============ NEW PARALLEL RULES FOR DETAILED ANALYSIS ============

rule prior_detailed_tfm:
    threads: 1
    conda: 'gretabench'
    input:
        grn=lambda wildcards: rules.grn_run.output.out.format(**wildcards),
        db='dbs/hg38/tfm/{db}/{db}.tsv',
    output:
        out='anl/metrics/prior_detailed/tfm/{db}/{dat}.{case}/{pre}.{p2g}.{tfb}.{mdl}.scores.csv',
        confusion='anl/metrics/prior_detailed/tfm/{db}/{dat}.{case}/{pre}.{p2g}.{tfb}.{mdl}.confusion.csv'
    params:
        subset=lambda wildcards: f'-s anl/metrics/prior_detailed/tfm/{wildcards.db}/{wildcards.dat}.{wildcards.case}.subset.csv' if wildcards.case != 'all' else ''
    shell:
        """
        python workflow/scripts/anl/metrics/prior/tfm_detailed.py \
        -a {input.grn} \
        -b {input.db} \
        -f {output.out} \
        -c {output.confusion} \
        {params.subset}
        """


rule prior_detailed_tfp:
    threads: 1
    conda: 'gretabench'
    input:
        grn=lambda wildcards: rules.grn_run.output.out.format(**wildcards),
        db='dbs/hg38/tfp/{db}/{db}.tsv',
    output:
        out='anl/metrics/prior_detailed/tfp/{db}/{dat}.{case}/{pre}.{p2g}.{tfb}.{mdl}.scores.csv',
        confusion='anl/metrics/prior_detailed/tfp/{db}/{dat}.{case}/{pre}.{p2g}.{tfb}.{mdl}.confusion.csv'
    params:
        thr_p=0.01,
    shell:
        """
        python workflow/scripts/anl/metrics/prior/tfp_detailed.py \
        -a {input.grn} \
        -b {input.db} \
        -p {params.thr_p} \
        -f {output.out} \
        -c {output.confusion}
        """


rule aggregate_tfm_confusion_detailed:
    threads: 1
    conda: 'gretabench'
    input:
        lambda w: [f.replace('.scores.csv', '.confusion.csv') for f in make_metric_inputs(w=w, mthds=mthds, baselines=baselines, rule_name='prior_detailed_tfm')]
    output:
        out='anl/metrics/prior_detailed/tfm/{db}/{dat}.{case}.confusion_agg.csv'
    shell:
        """
        python workflow/scripts/anl/metrics/aggregate_confusion.py \
        -i {input} \
        -o {output.out}
        """


rule aggregate_tfp_confusion_detailed:
    threads: 1
    conda: 'gretabench'
    input:
        lambda w: [f.replace('.scores.csv', '.confusion.csv') for f in make_metric_inputs(w=w, mthds=mthds, baselines=baselines, rule_name='prior_detailed_tfp')]
    output:
        out='anl/metrics/prior_detailed/tfp/{db}/{dat}.{case}.confusion_agg.csv'
    shell:
        """
        python workflow/scripts/anl/metrics/aggregate_confusion.py \
        -i {input} \
        -o {output.out}
        """


rule prior_c2g:
    threads: 1
    # singularity: 'workflow/envs/gretabench.sif'
    conda: 'gretabench'
    input:
        grn=lambda wildcards: rules.grn_run.output.out.format(**wildcards),
        resource='dbs/hg38/c2g/{db}/{db}.bed',
    output:
        out='anl/metrics/prior/c2g/{db}/{dat}.{case}/{pre}.{p2g}.{tfb}.{mdl}.scores.csv'
    params:
        grp='target',
    shell:
        """
        python workflow/scripts/anl/metrics/prior/gnm.py \
        -a {input.grn} \
        -b {input.resource} \
        -d {params.grp} \
        -f {output}
        """
