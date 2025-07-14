module.exports = (sequelize, DataTypes) => {
    const Patient = sequelize.define('Patient', {
        doctor: { type: DataTypes.STRING, allowNull: false },       // Doctor's Name
        patientName: { type: DataTypes.STRING, allowNull: false },  // Patient's Name
        age: { type: DataTypes.INTEGER, allowNull: false },         // Age
        gender: { type: DataTypes.STRING, allowNull: false },       // Gender
        contact: { type: DataTypes.STRING, allowNull: false },      // Contact
        medicalHistory: { type: DataTypes.TEXT },                   // Medical History
        medications: { type: DataTypes.TEXT },                      // Medications
        allergies: { type: DataTypes.TEXT },                        // Allergies
        diagnosis: { type: DataTypes.TEXT },                        // Diagnosis
        date: { type: DataTypes.DATEONLY, defaultValue: DataTypes.NOW } // Date
    });

    // If needed, associate with User (doctor model)
    Patient.associate = (models) => {
        Patient.belongsTo(models.User, { foreignKey: 'userId', as: 'doctorDetails' });
    };

    return Patient;
};

