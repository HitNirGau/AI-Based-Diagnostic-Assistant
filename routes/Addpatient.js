const express = require('express');
const router = express.Router();
const fs = require('fs');
const path = require('path');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

// Path to the CSV file
const csvFilePath = path.join(__dirname, '../data/patients.csv');

// Initialize CSV Writer
const csvWriter = createCsvWriter({
  path: csvFilePath,
  header: [
    { id: 'doctor', title: "Doctor's Name" },    // Matches the frontend field 'doctor'
    { id: 'patient', title: 'Patient Name' },    // Changed to match the form field 'patient'
    { id: 'age', title: 'Age' },
    { id: 'gender', title: 'Gender' },
    { id: 'contact', title: 'Contact' },
    { id: 'medicalHistory', title: 'Medical History' },
    { id: 'medications', title: 'Medications' },
    { id: 'allergies', title: 'Allergies' },
    { id: 'diagnosis', title: 'Diagnosis' },
    { id: 'date', title: 'Date' }
  ],
  append: fs.existsSync(csvFilePath)
});

// Add Patient Route
router.post('/addpatient', async (req, res) => {
  try {
    const patientData = {
      doctor: req.body.doctor,               // Matches form input
      patient: req.body.patient,             // Matches form input
      age: req.body.age,
      gender: req.body.gender,
      contact: req.body.contact,
      medicalHistory: req.body.history,      // Matches form input
      medications: req.body.medications,
      allergies: req.body.allergies,
      diagnosis: req.body.diagnosis,
      date: new Date().toISOString().split('T')[0] // Adds current date
    };

    console.log('Received Patient Data:', patientData);

    await csvWriter.writeRecords([patientData]);

    res.status(201).json({ message: '✅ Patient added successfully!', patient: patientData });
    
  } catch (error) {
    console.error('Error writing to CSV:', error);
    res.status(500).json({ error: '❌ Failed to add patient.' });
  }
});

module.exports = router;
