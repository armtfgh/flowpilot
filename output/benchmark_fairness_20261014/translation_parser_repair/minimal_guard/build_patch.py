from pathlib import Path
import difflib,hashlib
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3]
p=ROOT/'flora_translate/translation_llm.py';old=p.read_text()
needle='''        """Keep basic reactor-volume/flow/residence-time consistency."""
        def _unwrap(value, default=None):
'''
replacement='''        """Keep basic reactor-volume/flow/residence-time consistency."""
        # Validate the response boundary before filling optional field defaults.
        # An outer final-output envelope is not the internal FlowProposal schema.
        if not isinstance(data, dict) or not data:
            raise ValueError("FlowProposal must be a nonempty JSON object.")
        if "proposal" in data:
            raise ValueError(
                "Nested final-output 'proposal' envelope is not a FlowProposal. "
                "Return the scientific fields at the top level; retain any "
                "blocking reason in the flat safety_flags and reasoning fields."
            )
        if not set(data).intersection(FlowProposal.model_fields):
            raise ValueError("Response contains no recognized FlowProposal fields.")
        def _unwrap(value, default=None):
'''
assert old.count(needle)==1;new=old.replace(needle,replacement)
(BASE/'candidate_translation_llm.py').write_text(new)
(BASE/'reject_nonproposal_payload.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/flora_translate/translation_llm.py',tofile='b/flora_translate/translation_llm.py')))
(BASE/'production_translation_llm_sha256.txt').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')
print('Candidate only; production translation_llm.py untouched')
