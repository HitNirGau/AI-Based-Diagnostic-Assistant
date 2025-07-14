const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs');
const csv = require('csv-parser'); // Importing csv-parser
const csvFilePath = path.join(__dirname, '../data/patients.csv');

// Read CSV data and return the last 3 patients' history
router.get('/patients', (req, res) => {
    const patients = [];

    fs.createReadStream(csvFilePath)
        .pipe(csv())
        .on('data', (row) => {
            patients.push(row);  // ✅ Collect each row
        })
        .on('end', () => {
            res.json(patients);  // ✅ Send JSON response after reading the CSV
        })
        .on('error', (err) => {
            console.error('❌ CSV Read Error:', err);  // ✅ Log error for debugging
            res.status(500).json({ error: 'Failed to read patients data' });  // ✅ Send JSON error response
        });
});


// // Search patients by name
router.get('/patients/search', (req, res) => {
    console.log('🔍 Search request received:', req.query.name);  // Debug log

    const query = req.query.name ? req.query.name.toLowerCase() : '';
    const patients = [];

    fs.createReadStream(csvFilePath)
        .pipe(csv())
        .on('data', (row) => {
            patients.push(row);
        })
        .on('end', () => {
            console.log('✅ Patients loaded:', patients.length);
            const filteredPatients = patients.filter(patient =>
                patient['Patient Name'] && patient['Patient Name'].toLowerCase().includes(query)
            );
            console.log('🎯 Filtered patients:', filteredPatients.length);
            res.json(filteredPatients);
        })
        .on('error', (err) => {
            console.error('❌ Error reading CSV:', err);
            res.status(500).json({ error: 'Failed to read patients data' });
        });
});


router.get('/upload_xray', (req, res) => {
    console.log("xray");
    res.sendFile(path.join(__dirname, '../templates/upload_xray.html'));
});

module.exports = router;
