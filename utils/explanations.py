"""
Plain-language explanations for each modality × predicted class.
No jargon. No architecture references.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class Explanation:
    title: str
    summary: str                   # 1–2 lines
    key_findings: List[str]        # 3 bullets max
    meaning: str                   # 2 lines max — what this means for the patient
    confidence_note: str           # one plain sentence
    next_step: str                 # one actionable sentence


CT_EXPLANATIONS = {
    "Adenocarcinoma": Explanation(
        title="Possible Adenocarcinoma",
        summary="The scan shows patterns commonly associated with adenocarcinoma, "
                "the most common type of lung cancer.",
        key_findings=[
            "Irregular tissue growth detected in the outer lung area",
            "Ground-glass texture present — a recognised early sign",
            "No clear sign of spread to surrounding structures",
        ],
        meaning="Adenocarcinoma typically grows slowly and is often caught before "
                "it has spread. Early-stage cases generally have good treatment options.",
        confidence_note="This is a screening result, not a diagnosis.",
        next_step="Refer to a pulmonologist or oncologist for biopsy and confirmed staging.",
    ),
    "Large Cell Carcinoma": Explanation(
        title="Possible Large Cell Carcinoma",
        summary="The scan shows a large, irregular mass with features that suggest "
                "an aggressive lung cancer type.",
        key_findings=[
            "Large mass detected with irregular, poorly-defined edges",
            "Areas of possible central breakdown (necrosis)",
            "Located in the peripheral lung region",
        ],
        meaning="Large cell carcinoma tends to grow quickly. "
                "Early confirmation and staging are important for treatment planning.",
        confidence_note="This finding needs histological confirmation before any clinical action.",
        next_step="Urgent referral to oncology for tissue biopsy and staging workup.",
    ),
    "Squamous Cell Carcinoma": Explanation(
        title="Possible Squamous Cell Carcinoma",
        summary="The scan shows a central lung mass with features typical of "
                "squamous cell carcinoma.",
        key_findings=[
            "Mass detected near the central airways",
            "Possible cavitation (hollow area) within the mass",
            "Associated changes in surrounding lung tissue",
        ],
        meaning="Squamous cell carcinoma arising centrally can sometimes be reached "
                "via bronchoscopy, which may simplify the biopsy process.",
        confidence_note="Findings are indicative only — tissue sampling is required.",
        next_step="Refer to pulmonology for bronchoscopy assessment and biopsy.",
    ),
    "Normal": Explanation(
        title="No Significant Finding",
        summary="The scan does not show features associated with the lung cancer "
                "types in this system's scope.",
        key_findings=[
            "No suspicious mass or nodule identified",
            "No unusual tissue density patterns detected",
            "Lung fields appear within normal range",
        ],
        meaning="A normal result is reassuring. However, this tool only screens "
                "for four specific conditions — it cannot rule out all lung diseases.",
        confidence_note="This system does not replace a full radiological review.",
        next_step="Continue routine screening as advised by your physician.",
    ),
}

US_EXPLANATIONS = {
    "Benign": Explanation(
        title="Likely Benign Lesion",
        summary="The scan shows features more commonly seen in non-cancerous breast "
                "masses such as cysts or fibroadenomas.",
        key_findings=[
            "Well-defined, smooth edges around the lesion",
            "Brightness pattern behind the mass suggests fluid content",
            "No irregular internal structure detected",
        ],
        meaning="Most benign breast masses do not require immediate treatment. "
                "They are monitored with routine follow-up imaging.",
        confidence_note="A benign classification does not fully exclude malignancy — "
                        "clinical correlation is needed.",
        next_step="Follow up with your doctor or a breast specialist as part of routine care.",
    ),
    "Malignant": Explanation(
        title="Features Suggest Malignancy",
        summary="The scan shows acoustic and structural patterns that are more "
                "commonly associated with malignant breast masses.",
        key_findings=[
            "Shadow behind the mass — a key sign of dense, cancerous tissue",
            "Irregular or ill-defined edges around the lesion",
            "Heterogeneous internal texture detected",
        ],
        meaning="These features are associated with malignancy, but a scan alone "
                "cannot confirm cancer. A biopsy is required for certainty.",
        confidence_note="This is a screening flag, not a cancer diagnosis.",
        next_step="See a breast specialist promptly for clinical examination and biopsy.",
    ),
    "Normal": Explanation(
        title="No Lesion Detected",
        summary="The scan does not show any identifiable mass or structural abnormality "
                "in the breast tissue.",
        key_findings=[
            "No focal mass or nodule identified",
            "Tissue echogenicity appears uniform",
            "No suspicious acoustic patterns present",
        ],
        meaning="Normal ultrasound findings are reassuring. Ultrasound alone may "
                "miss small or early-stage lesions.",
        confidence_note="This system screens for three categories only and is not a "
                        "complete breast assessment.",
        next_step="Continue age-appropriate breast screening as recommended by your doctor.",
    ),
}

MRI_EXPLANATIONS = {
    "Glioma": Explanation(
        title="Possible Glioma",
        summary="The MRI shows a brain mass with irregular borders and internal variation "
                "consistent with a glioma.",
        key_findings=[
            "Irregular mass with infiltrative-looking edges",
            "Possible areas of internal breakdown or necrosis",
            "Surrounding tissue shows signs of involvement",
        ],
        meaning="Gliomas range widely in severity. Grading requires a tissue biopsy. "
                "Prompt specialist assessment is important.",
        confidence_note="MRI patterns are indicative — histological grading is essential.",
        next_step="Urgent referral to neurology or neurosurgery for biopsy and grading.",
    ),
    "Meningioma": Explanation(
        title="Possible Meningioma",
        summary="The MRI shows a well-defined mass along the brain's outer lining, "
                "consistent with a meningioma.",
        key_findings=[
            "Compact, well-defined mass attached to the brain surface",
            "Homogeneous internal texture — a sign of slower growth",
            "No obvious infiltration into brain tissue",
        ],
        meaning="Most meningiomas are non-cancerous and grow slowly. "
                "Many are monitored without immediate surgery.",
        confidence_note="Appearance is characteristic but confirmation requires specialist review.",
        next_step="Refer to neurosurgery for assessment. Many cases are managed conservatively.",
    ),
    "Pituitary": Explanation(
        title="Possible Pituitary Tumour",
        summary="The MRI shows a mass in the pituitary region at the base of the brain.",
        key_findings=[
            "Small, defined mass in the sellar region",
            "Located near the optic pathway — may affect vision",
            "No obvious spread beyond the pituitary area",
        ],
        meaning="Pituitary tumours are usually benign and often hormone-secreting. "
                "Symptoms may include hormonal changes or visual disturbance.",
        confidence_note="Size and hormonal activity determine management — specialist review needed.",
        next_step="Refer to endocrinology and neurosurgery for hormone testing and further assessment.",
    ),
    "No Tumor": Explanation(
        title="No Tumour Detected",
        summary="The MRI does not show features associated with the brain tumour "
                "types in this system's scope.",
        key_findings=[
            "No identifiable mass in the brain parenchyma",
            "No abnormal signal indicating glioma, meningioma, or pituitary lesion",
            "Brain structures appear symmetrical",
        ],
        meaning="This is a reassuring result. However, this system only screens "
                "for three specific tumour types and cannot exclude all brain conditions.",
        confidence_note="A normal result here does not constitute a clear radiological report.",
        next_step="If symptoms persist, speak with a neurologist for a full clinical MRI review.",
    ),
}

ALL_EXPLANATIONS = {
    "CT Scan":    CT_EXPLANATIONS,
    "Ultrasound": US_EXPLANATIONS,
    "MRI":        MRI_EXPLANATIONS,
}

def get_explanation(modality: str, predicted_class: str) -> Explanation:
    return ALL_EXPLANATIONS[modality][predicted_class]
