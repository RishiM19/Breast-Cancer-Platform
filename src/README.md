# Lumina Breast AI

**Illuminating the path to precise diagnosis**

A medical software application for AI-assisted breast cancer diagnosis with a clinical, minimalist design.

## Features

- 🔐 **Login System** - Demo credentials: `demo@lumina.ai` / `demo123`
- 📊 **Dashboard** - View stats, upload scans, and manage patient cases
- 🤖 **AI Analysis** - Automated diagnostic analysis with confidence scores
- 📋 **Patient Database** - Search, filter, and manage patient records
- 📄 **Diagnostic Reports** - Detailed AI analysis reports with export/print functionality
- ✏️ **Correction Mode** - Professional medical image annotation interface
- ⚙️ **Settings** - Customize user preferences
- 🔔 **Notifications** - Stay updated with system alerts

## Prerequisites

Before running this application, make sure you have the following installed:

- **Node.js** (version 18 or higher) - [Download here](https://nodejs.org/)
- **npm** (comes with Node.js) or **yarn**

## Installation & Setup

Follow these steps to run the application on your laptop:

### 1. Download the Project Files

Download all the project files to a folder on your laptop. Make sure you have the following structure:

```
lumina-breast-ai/
├── App.tsx
├── main.tsx
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tsconfig.node.json
├── components/
│   ├── Dashboard.tsx
│   ├── AnalysisResults.tsx
│   ├── CorrectionMode.tsx
│   ├── PatientList.tsx
│   ├── DiagnosticReport.tsx
│   ├── Settings.tsx
│   ├── ModelStats.tsx
│   └── figma/
│       └── ImageWithFallback.tsx
└── styles/
    └── globals.css
```

### 2. Open Terminal/Command Prompt

- **Windows**: Press `Win + R`, type `cmd`, and press Enter
- **Mac**: Press `Cmd + Space`, type `terminal`, and press Enter
- **Linux**: Press `Ctrl + Alt + T`

### 3. Navigate to Project Folder

In the terminal, navigate to the folder where you saved the project:

```bash
cd path/to/lumina-breast-ai
```

For example:
```bash
cd C:\Users\YourName\Documents\lumina-breast-ai
```

### 4. Install Dependencies

Run the following command to install all required packages:

```bash
npm install
```

This will download and install all the necessary libraries (React, TypeScript, Tailwind CSS, etc.)

**Note**: This step might take a few minutes depending on your internet connection.

### 5. Start the Development Server

Once installation is complete, start the application:

```bash
npm run dev
```

### 6. Open in Browser

The application should automatically open in your default browser at:

```
http://localhost:3000
```

If it doesn't open automatically, manually open your browser and go to that URL.

## Login Credentials

Use these demo credentials to log in:

- **Email**: `demo@lumina.ai`
- **Password**: `demo123`

## Usage

1. **Login** - Use the demo credentials to access the dashboard
2. **Upload Scans** - Click the upload area to simulate uploading medical scans
3. **View Dashboard** - See patient stats and recent cases
4. **Patient List** - Click "Patient Database" in the sidebar to view all patients
5. **Analyze Scans** - Click "Analyze" on any patient to generate AI diagnostic results
6. **View Reports** - After analysis, click "View Report" to see detailed results
7. **Settings** - Customize your profile and preferences

## Available Commands

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Troubleshooting

### Port Already in Use

If port 3000 is already in use, the application will automatically try to use another port (usually 3001, 3002, etc.). Check the terminal output for the correct URL.

### Installation Errors

If you encounter errors during `npm install`, try:

1. Delete the `node_modules` folder and `package-lock.json` file
2. Run `npm install` again
3. Make sure you have Node.js version 18 or higher: `node --version`

### Cannot Find Module Errors

Make sure all files are in the correct folder structure as shown above.

## Technology Stack

- **React 18** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool and dev server
- **Tailwind CSS 4** - Styling
- **Motion (Framer Motion)** - Animations
- **Lucide React** - Icons

## Design System

- **Primary Color**: #007AFF (Medical Blue)
- **Background**: White with glassmorphism effects
- **Typography**: Clean, modern sans-serif
- **Style**: Clinical, minimalist, professional

## Notes

⚠️ **Important**: This is a demo application with simulated data. Do not use for actual medical diagnosis or patient data storage.

## Support

For issues or questions, please refer to the project documentation or contact the development team.

---

Made with ❤️ for healthcare professionals
