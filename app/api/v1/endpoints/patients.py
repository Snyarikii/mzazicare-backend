"""Authenticated synthetic geriatric care profiles for the mobile workflow.

The repository currently has no patient table or patient data source. These
profiles are demo fixtures and must not be used to direct real clinical care.
"""
from copy import deepcopy
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_current_active_user
from app.models.entities import User

router = APIRouter()

# Synthetic Kenyan geriatric-care demo records; no real patient information.
_PATIENTS: list[dict[str, Any]] = [
    {
        "id": 1, "full_name": "Mzee Juma Bakari", "age": 74,
        "chronic_conditions": ["Type 2 Diabetes", "Hypertension", "Mobility Impairment"],
        "care_tier": "High Dependency", "assigned_family_member": "Asha Bakari (daughter)",
        "daily_vitals": {"blood_pressure": "148/86 mmHg", "pulse": "78 bpm", "blood_glucose": "8.4 mmol/L"},
        "last_care_notes": "Reports mild knee pain when standing. Used walking frame indoors; meals and fluids taken well. Family asked to keep pathway clear and report dizziness or a fall.",
        "clinical_file": {
            "file_number": "GER-DEMO-1001", "record_status": "Active synthetic demo record",
            "reviewed_on": "2026-09-20", "primary_clinician": "Community Geriatric Team A",
            "medical_history": [
                {"condition": "Type 2 Diabetes", "onset_year": 2014, "status": "Managed with oral medication and meal planning", "monitoring": "Fasting glucose log reviewed weekly"},
                {"condition": "Hypertension", "onset_year": 2011, "status": "Ongoing", "monitoring": "Home blood pressure morning and evening"},
                {"condition": "Mobility impairment", "onset_year": 2022, "status": "Uses walking frame; needs standby support outdoors", "monitoring": "Assess transfers and fall risk at each visit"},
            ],
            "allergies": [{"substance": "Sulfonamides", "reaction": "Rash", "severity": "Moderate"}],
            "functional_assessment": {"mobility": "Walking frame; standby assistance", "bathing": "Setup and supervision", "dressing": "Independent with extra time", "feeding": "Independent", "continence": "Occasional urgency", "cognition": "Alert and oriented; no change reported", "fall_risk": "High"},
            "care_goals": ["Maintain safe transfers and indoor mobility", "Keep glucose and blood pressure within clinician-defined targets", "Support medication adherence and regular meals", "Prevent skin breakdown with daily inspection"],
            "nutrition_and_hydration": "Regular meals with consistent carbohydrate portions; offer water with medicines and between meals. No swallowing difficulty reported.",
        },
        "medication_administration_schedule": [
            {"medicine": "Metformin", "dose": "500 mg", "route": "Oral", "time": "08:00 and 20:00", "instructions": "Give with meals; record dose and tolerance.", "administered_by": "Family caregiver", "status": "Scheduled"},
            {"medicine": "Amlodipine", "dose": "5 mg", "route": "Oral", "time": "08:00", "instructions": "Check and record morning blood pressure before dose; follow clinician hold instructions.", "administered_by": "Family caregiver", "status": "Scheduled"},
            {"medicine": "Vitamin D3", "dose": "1000 IU", "route": "Oral", "time": "13:00", "instructions": "Give with lunch.", "administered_by": "Family caregiver", "status": "Scheduled"},
        ],
        "caregiver_visit_logs": [
            {"date": "2026-09-20", "time": "09:15", "caregiver": "Lydia Naliaka", "visit_type": "Home wellness visit", "observations": "BP 148/86 mmHg; glucose 8.4 mmol/L before breakfast. No acute symptoms.", "care_provided": "Medication reconciliation, supervised transfer practice, hydration reminder and skin check.", "follow_up": "Repeat BP this evening; clinician review if readings remain above the home plan."},
            {"date": "2026-09-17", "time": "10:00", "caregiver": "Lydia Naliaka", "visit_type": "Mobility and safety review", "observations": "No falls since prior visit. Mild knee discomfort on standing.", "care_provided": "Checked walking frame and clear routes; practiced sit-to-stand with standby assistance.", "follow_up": "Family to report worsening pain, new weakness or any fall."},
            {"date": "2026-09-13", "time": "08:45", "caregiver": "Peter Mwangi", "visit_type": "Medication support", "observations": "Medication pack consistent with schedule; appetite stable.", "care_provided": "Reviewed dose log and glucose diary with family caregiver.", "follow_up": "Continue daily log and bring it to next community review."},
            {"date": "2026-09-10", "time": "11:20", "caregiver": "Lydia Naliaka", "visit_type": "Routine home visit", "observations": "Alert, comfortable at rest; no skin concerns noted.", "care_provided": "Assisted with personal care setup and checked footwear fit.", "follow_up": "Routine visit planned in three days."},
        ],
        "emergency_contacts": [
            {"name": "Asha Bakari", "relationship": "Daughter and primary family contact", "phone": "+254 700 555 101", "priority": 1},
            {"name": "Salim Bakari", "relationship": "Son", "phone": "+254 711 555 101", "priority": 2},
        ],
        "emergency_guidance": "For collapse, severe breathing difficulty, chest pain, new facial droop, sudden weakness or serious injury, contact local emergency services immediately and notify the family contact. This demo file is not an emergency triage tool.",
        "next_review": "2026-09-27",
    },
    {
        "id": 2, "full_name": "Mama Grace Wanjiku", "age": 68,
        "chronic_conditions": ["Osteoarthritis", "Hypertension"],
        "care_tier": "Daily Assistance", "assigned_family_member": "Peter Wanjiku (son)",
        "daily_vitals": {"blood_pressure": "136/80 mmHg", "pulse": "72 bpm", "blood_glucose": "5.9 mmol/L"},
        "last_care_notes": "Comfortable after morning routine. Reports hand stiffness that eases with gentle movement. Continue safe bathroom transfers and encourage short walks as tolerated.",
        "clinical_file": {
            "file_number": "GER-DEMO-1002", "record_status": "Active synthetic demo record", "reviewed_on": "2026-09-19", "primary_clinician": "Community Geriatric Team B",
            "medical_history": [{"condition": "Osteoarthritis", "onset_year": 2018, "status": "Hand and knee stiffness", "monitoring": "Pain, function and response to prescribed plan"}, {"condition": "Hypertension", "onset_year": 2016, "status": "Stable at recent home checks", "monitoring": "Weekly blood pressure review"}],
            "allergies": [], "functional_assessment": {"mobility": "Independent indoors; cane for longer walks", "bathing": "Standby assistance", "dressing": "Independent", "feeding": "Independent", "continence": "Independent", "cognition": "Alert; family reports no concerns", "fall_risk": "Moderate"},
            "care_goals": ["Preserve independence in personal care", "Support comfortable daily movement", "Maintain blood pressure monitoring", "Reduce household trip hazards"],
            "nutrition_and_hydration": "Balanced meals; encourage regular fluids, especially during warm weather. No dietary restrictions recorded in this synthetic file.",
        },
        "medication_administration_schedule": [{"medicine": "Lisinopril", "dose": "10 mg", "route": "Oral", "time": "08:00", "instructions": "Follow current clinician instructions and record dose.", "administered_by": "Self with family check", "status": "Scheduled"}, {"medicine": "Paracetamol", "dose": "500 mg", "route": "Oral", "time": "As needed", "instructions": "Use only as prescribed; record use and report persistent pain.", "administered_by": "Family caregiver", "status": "As needed"}],
        "caregiver_visit_logs": [{"date": "2026-09-19", "time": "08:30", "caregiver": "Mercy Atieno", "visit_type": "Daily living support", "observations": "Hand stiffness in the morning; no new swelling or injury reported.", "care_provided": "Bathroom safety check, meal setup and gentle range-of-motion reminders.", "follow_up": "Review comfort and walking tolerance at next visit."}, {"date": "2026-09-18", "time": "08:40", "caregiver": "Mercy Atieno", "visit_type": "Wellness check", "observations": "BP 136/80 mmHg; steady gait indoors.", "care_provided": "Medication prompt and short supervised walk in the compound.", "follow_up": "Continue routine checks; family to report a fall or sudden change."}, {"date": "2026-09-17", "time": "09:00", "caregiver": "Peter Wanjiku", "visit_type": "Family check-in", "observations": "Meals and sleep reported as usual.", "care_provided": "Prepared weekly medicine reminder and checked bathroom handrail.", "follow_up": "Care worker visit scheduled next morning."}],
        "emergency_contacts": [{"name": "Peter Wanjiku", "relationship": "Son and primary family contact", "phone": "+254 700 555 102", "priority": 1}, {"name": "Lydia Wanjiku", "relationship": "Niece", "phone": "+254 711 555 102", "priority": 2}],
        "emergency_guidance": "For a fall with injury, sudden confusion, chest pain or difficulty breathing, seek urgent local medical help and notify the family contact. This synthetic record is not a substitute for clinical advice.", "next_review": "2026-09-26",
    },
    {
        "id": 3, "full_name": "David Kiprono", "age": 81,
        "chronic_conditions": ["Chronic Obstructive Pulmonary Disease", "Type 2 Diabetes", "Hearing Impairment"],
        "care_tier": "High Dependency", "assigned_family_member": "Ruth Kiprono (niece)",
        "daily_vitals": {"blood_pressure": "130/78 mmHg", "pulse": "84 bpm", "blood_glucose": "7.2 mmol/L"},
        "last_care_notes": "Breathing comfortable at rest with usual inhaler plan. Uses hearing aid; speak face-to-face and confirm understanding. No recent fall reported.",
        "clinical_file": {"file_number": "GER-DEMO-1003", "record_status": "Active synthetic demo record", "reviewed_on": "2026-09-21", "primary_clinician": "Community Geriatric Team A", "medical_history": [{"condition": "COPD", "onset_year": 2012, "status": "Stable on prescribed inhaler plan", "monitoring": "Breathlessness, cough, sputum and oxygen symptoms"}, {"condition": "Type 2 Diabetes", "onset_year": 2017, "status": "Oral treatment", "monitoring": "Glucose log and meal intake"}, {"condition": "Hearing impairment", "onset_year": 2020, "status": "Uses hearing aid", "monitoring": "Device function and communication needs"}], "allergies": [{"substance": "No known drug allergies", "reaction": "None reported", "severity": "Not applicable"}], "functional_assessment": {"mobility": "Cane indoors; supervision outdoors", "bathing": "Assistance getting in and out of shower", "dressing": "Prompting and occasional help", "feeding": "Independent", "continence": "Independent", "cognition": "Oriented; allow time for communication", "fall_risk": "Moderate"}, "care_goals": ["Support breathing comfort and prescribed inhaler adherence", "Maintain glucose and meal routine", "Use accessible communication", "Keep walkways clear and reduce exertion during care"], "nutrition_and_hydration": "Small regular meals and fluids through the day. Pause activity if breathless and follow the clinician's established action plan."},
        "medication_administration_schedule": [{"medicine": "Metformin", "dose": "500 mg", "route": "Oral", "time": "08:00 and 20:00", "instructions": "Give with meals and record administration.", "administered_by": "Family caregiver", "status": "Scheduled"}, {"medicine": "Tiotropium inhaler", "dose": "18 mcg", "route": "Inhaled", "time": "08:00", "instructions": "Use according to prescribed technique; verify device dose indicator.", "administered_by": "Self with caregiver prompt", "status": "Scheduled"}, {"medicine": "Salbutamol inhaler", "dose": "As prescribed", "route": "Inhaled", "time": "As needed", "instructions": "Use only as directed in the existing clinician action plan; record use.", "administered_by": "Self or caregiver", "status": "As needed"}],
        "caregiver_visit_logs": [{"date": "2026-09-21", "time": "10:10", "caregiver": "Peter Mwangi", "visit_type": "Respiratory wellness check", "observations": "Comfortable at rest; usual cough, no new fever reported.", "care_provided": "Confirmed inhaler supplies, checked hearing aid battery and reviewed symptom log.", "follow_up": "Family to seek clinical advice for worsening breathlessness or change in sputum."}, {"date": "2026-09-18", "time": "09:30", "caregiver": "Ruth Kiprono", "visit_type": "Family medication support", "observations": "Breakfast taken; glucose 7.2 mmol/L on home meter.", "care_provided": "Observed inhaler routine and prepared accessible written reminders.", "follow_up": "Continue usual schedule and bring logs to next review."}, {"date": "2026-09-15", "time": "10:00", "caregiver": "Peter Mwangi", "visit_type": "Home safety review", "observations": "No falls; mild fatigue after longer walk.", "care_provided": "Cleared corridor, checked cane tip and reviewed rest breaks.", "follow_up": "Use supervision outdoors and pace activity."}],
        "emergency_contacts": [{"name": "Ruth Kiprono", "relationship": "Niece and primary family contact", "phone": "+254 700 555 103", "priority": 1}, {"name": "James Kiprono", "relationship": "Brother", "phone": "+254 711 555 103", "priority": 2}],
        "emergency_guidance": "For severe or sudden breathing difficulty, blue lips, collapse, chest pain or new one-sided weakness, contact local emergency services and notify the family contact. Follow the clinician's written action plan. This synthetic file is not an emergency triage tool.", "next_review": "2026-09-28",
    },
    {
        "id": 4, "full_name": "Mama Beatrice Muthoni", "age": 85,
        "chronic_conditions": ["Heart Failure", "Hypertension", "Osteoporosis"],
        "care_tier": "Daily Assistance", "assigned_family_member": "Mary Muthoni (granddaughter)",
        "daily_vitals": {"blood_pressure": "142/82 mmHg", "pulse": "76 bpm", "blood_glucose": "6.1 mmol/L"},
        "last_care_notes": "Mild ankle swelling unchanged from baseline per family. Rest periods offered during dressing. Continue daily weight log as directed by her care team.",
        "clinical_file": {"file_number": "GER-DEMO-1004", "record_status": "Active synthetic demo record", "reviewed_on": "2026-09-18", "primary_clinician": "Community Geriatric Team B", "medical_history": [{"condition": "Heart failure", "onset_year": 2019, "status": "Managed with clinician-directed regimen", "monitoring": "Breathlessness, swelling, and daily weight trend"}, {"condition": "Hypertension", "onset_year": 2008, "status": "Ongoing", "monitoring": "Home blood pressure readings"}, {"condition": "Osteoporosis", "onset_year": 2015, "status": "Fall prevention plan in place", "monitoring": "Pain or suspected injury after a fall"}], "allergies": [], "functional_assessment": {"mobility": "Walker indoors with supervision", "bathing": "Hands-on assistance", "dressing": "Help with lower garments", "feeding": "Independent after setup", "continence": "Uses bathroom with assistance", "cognition": "Alert; family reports usual memory", "fall_risk": "High"}, "care_goals": ["Monitor for changes in breathing or swelling", "Support safe daily personal care", "Maintain a daily weight and symptom record", "Prevent falls and bone injury"], "nutrition_and_hydration": "Follow existing clinician guidance for salt and fluids; do not independently change fluid limits. Offer meals seated and allow rest breaks."},
        "medication_administration_schedule": [{"medicine": "Furosemide", "dose": "20 mg", "route": "Oral", "time": "08:00", "instructions": "Follow current prescribed plan; record dose and any concerns.", "administered_by": "Family caregiver", "status": "Scheduled"}, {"medicine": "Enalapril", "dose": "5 mg", "route": "Oral", "time": "08:00 and 20:00", "instructions": "Administer as prescribed; record dose.", "administered_by": "Family caregiver", "status": "Scheduled"}, {"medicine": "Calcium and Vitamin D", "dose": "As prescribed", "route": "Oral", "time": "13:00", "instructions": "Give with lunch according to labeled prescription.", "administered_by": "Family caregiver", "status": "Scheduled"}],
        "caregiver_visit_logs": [{"date": "2026-09-18", "time": "09:00", "caregiver": "Mercy Atieno", "visit_type": "Daily assistance visit", "observations": "Ankle swelling reported unchanged; no breathlessness at rest. Weight recorded in home log.", "care_provided": "Assisted dressing, checked footwear and supported walker transfer.", "follow_up": "Continue daily weight. Report rapid change or worsening symptoms to care team."}, {"date": "2026-09-17", "time": "09:10", "caregiver": "Mary Muthoni", "visit_type": "Family wellness check", "observations": "Rested comfortably overnight; appetite usual.", "care_provided": "Medication prompt, breakfast setup and clear pathway check.", "follow_up": "Caregiver to review symptoms at next scheduled visit."}, {"date": "2026-09-16", "time": "08:50", "caregiver": "Mercy Atieno", "visit_type": "Mobility support", "observations": "Steady transfer with walker and hands-on support.", "care_provided": "Supported wash and dressing with rest pauses; no skin concerns noted.", "follow_up": "Maintain supervised transfers and report any fall promptly."}],
        "emergency_contacts": [{"name": "Mary Muthoni", "relationship": "Granddaughter and primary family contact", "phone": "+254 700 555 104", "priority": 1}, {"name": "Joseph Muthoni", "relationship": "Son", "phone": "+254 711 555 104", "priority": 2}],
        "emergency_guidance": "For sudden or severe breathlessness, chest pain, fainting, confusion or a fall with suspected injury, contact local emergency services and notify family. Follow the current clinician care plan. This synthetic file is not an emergency triage tool.", "next_review": "2026-09-25",
    },
]

_BY_ID = {patient["id"]: patient for patient in _PATIENTS}


@router.get("", summary="List geriatric patient records")
async def list_patients(current_user: User = Depends(get_current_active_user)) -> list[dict[str, Any]]:
    """Return synthetic elderly care profiles for the authenticated workflow."""
    del current_user
    fields = ("id", "full_name", "age", "chronic_conditions", "care_tier",
              "assigned_family_member", "daily_vitals", "last_care_notes")
    return [{field: deepcopy(profile[field]) for field in fields} for profile in _PATIENTS]


@router.get("/{patient_id}", summary="Get a detailed geriatric care profile")
async def get_patient(patient_id: int,
                      current_user: User = Depends(get_current_active_user)) -> dict[str, Any]:
    """Return a synthetic geriatric clinical file, or 404 for an unknown identifier."""
    del current_user
    patient = _BY_ID.get(patient_id)
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    return deepcopy(patient)
