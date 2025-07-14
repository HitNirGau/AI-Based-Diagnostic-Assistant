// const createCsvWriter = require('csv-writer').createObjectCsvWriter;

// const csvWriter = createCsvWriter({
//   path: 'users.csv',
//   header: [
//     { id: 'userid', title: 'UserID' },
//     { id: 'username', title: 'Username' },
//     { id: 'password', title: 'Password' }
//   ]
// });

// // Sample user data
// const users = [
//   { userid: 1, username: 'user_1', password: '57210b12af5e06ad2e6e54a93b1465aa' },
//   { userid: 2, username: 'user_2', password: '259640f97ac2b4379dd540ff4016654c' },
//   { userid: 3, username: 'user_3', password: '48ef85c894a06a4562268de8e4d934e1' }
// ];

// // Write to CSV
// csvWriter.writeRecords(users)
//   .then(() => {
//     console.log('✅ User data saved to users.csv');
//   })
//   .catch((err) => {
//     console.error('❌ Error writing CSV:', err);
//   });


const fs = require('fs');
const { Parser } = require('json2csv');

function savePatientData(doctor, patient, diagnosis, imagePath) {
    const data = [{
        Doctor_ID: doctor.id,
        Doctor_Name: doctor.name,
        Patient_ID: patient.id,
        Patient_Name: patient.name,
        Age: patient.age,
        Gender: patient.gender,
        Contact: patient.contact,
        Address: patient.address,
        Medical_History: patient.medicalHistory.join(", "),
        Family_History: patient.familyHistory.join(", "),
        Medications: patient.medications.join(", "),
        Allergies: patient.allergies.join(", "),
        Diagnosis: diagnosis,
        Image_Path: imagePath,
        Date: new Date().toISOString().split('T')[0]
    }];

    const csv = new Parser({ header: false }).parse(data);
    fs.appendFileSync('patient_records.csv', `\n${csv}`);
}

// Example Usage
const doctor = { id: 'D001', name: 'Dr. Smith' };
const patient = {
    id: 'P004',
    name: 'Mark Davis',
    age: 55,
    gender: 'Male',
    contact: '9988776655',
    address: '101 Maple Ave',
    medicalHistory: ['Hypertension'],
    familyHistory: ['Colon Cancer (Father)'],
    medications: ['Lisinopril'],
    allergies: ['None']
};

savePatientData(doctor, patient, 'Malignant Tumor', '/images/mark_davis_mammo4.png');
