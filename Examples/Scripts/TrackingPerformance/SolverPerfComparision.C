// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <TROOT.h>

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../../../ATLAS_Plot/AtlasStyle.C"

/// @brief Set the range of the TEfficiency histogram
/// @param plot 
/// @param min 
/// @param max 
void setRange(TEfficiency *plot, float ymin, float ymax, float xmin, float xmax){
    gPad->Update(); 
    auto graph = plot->GetPaintedGraph(); 
    graph->SetMinimum(ymin);
    graph->SetMaximum(ymax);
    graph->GetXaxis()->SetRangeUser(xmin,xmax);
    gPad->Update();
}


void overlapHistograms(TFile *fCKF, TFile *fgreedySolver, TFile *fMLSolver, std::string histogramName, std::string variableName, float rangeMin, float rangeMax, float xMin, float xMax, TLegend* leg, std::string saveName){

    // Plot the histograme vs variableName
    std::string title = histogramName + "_vs_" + variableName;

    TCanvas *C = new TCanvas(title.c_str(), title.c_str());

    TEfficiency * hCKF = new TEfficiency(("CKF"+title).c_str(),("CKF"+title).c_str() , 100, 0, 100, 100, 0, 1);
    hCKF = (TEfficiency*)fCKF->Get((title).c_str());
    hCKF->SetLineColor(kBlack);
    hCKF->SetMarkerColor(kBlack);
    hCKF->SetMarkerStyle(20);
    hCKF->SetMarkerSize(0.8);
    hCKF->Draw("");
    setRange(hCKF, rangeMin, rangeMax, xMin, xMax);


    TEfficiency * hgreedySolver = new TEfficiency(("greedySolver"+title).c_str(),("greedySolver"+title).c_str() , 100, 0, 100, 100, 0, 1);
    hgreedySolver = (TEfficiency*)fgreedySolver->Get((title).c_str());
    hgreedySolver->SetLineColor(kOrange-3);
    hgreedySolver->SetMarkerColor(kOrange-3);
    hgreedySolver->SetMarkerStyle(21);
    hgreedySolver->SetMarkerSize(0.8);
    hgreedySolver->Draw("SAME");

    TEfficiency *hMLSolver = new TEfficiency(("MLSolver" + title).c_str(),
                                             ("MLSolver" + title).c_str(),
                                             100, 0, 100, 100, 0, 1);
    hMLSolver = (TEfficiency*)fMLSolver->Get((title).c_str());
    hMLSolver->SetLineColor(kAzure+5);
    hMLSolver->SetMarkerColor(kAzure+5);
    hMLSolver->SetMarkerStyle(22);
    hMLSolver->SetMarkerSize(0.8);
    hMLSolver->Draw("SAME");

    if(histogramName == "duplicationRate"){
        gPad->SetLogy();
        gPad->Update(); 
    }

    leg->AddEntry(hCKF,"CKF","lep");
    leg->AddEntry(hgreedySolver,"greedySolver","lep");
    leg->AddEntry(hMLSolver, "MLSolver", "lep");
    leg->Draw();

    C->SaveAs((saveName).c_str());

}

/// @brief Root macro to compare the performance of the different solvers
/// @param CKF path to the CKF output file
/// @param GreedySolver path to the greedySolver output file
/// @param MLSolver path to the MLSolver output file
void SolverPerfComparision(std::string CKF = "", std::string greedySolver = "", std::string MLSolver = ""){

    // Set the ATLAS style
  gROOT->LoadMacro("AtlasLabels.C");
  SetAtlasStyle();
  gStyle->SetErrorX(0);

  // Open the files
  TFile *fCKF = new TFile(CKF.c_str());
  TFile *fgreedySolver = new TFile(greedySolver.c_str());
  TFile *fMLSolver = new TFile(MLSolver.c_str());

  // Plot the track efficiency
  TLegend *leg_eff_pt = new TLegend(0.65, 0.20, 0.92, 0.35);
  overlapHistograms(fCKF, fgreedySolver, fMLSolver, "trackeff", "pT", 0.4, 1.05, 0.0, 60.0,
                    leg_eff_pt, "plot/trackEfficiency_pt.pdf");
  TLegend *leg_eff_eta = new TLegend(0.65, 0.25, 0.92, 0.40);
  overlapHistograms(fCKF, fgreedySolver, fMLSolver, "trackeff", "eta", 0.4, 1.05, -3.2, 3.2,
                    leg_eff_eta, "plot/trackEfficiency_eta.pdf");
  TLegend *leg_eff_phi = new TLegend(0.65, 0.25, 0.92, 0.40);
  overlapHistograms(fCKF, fgreedySolver, fMLSolver, "trackeff", "phi", 0.8, 1.05, -3.2, 3.2,
                    leg_eff_phi, "plot/trackEfficiency_phi.pdf");
  TLegend *leg_eff_DeltaR = new TLegend(0.65, 0.25, 0.92, 0.40);
  overlapHistograms(fCKF, fgreedySolver, fMLSolver, "trackeff", "DeltaR", 0.8, 1.05, 0.0, 0.3,
                    leg_eff_DeltaR, "plot/trackEfficiency_deltaR.pdf");

  // Plot the fake rate
  TLegend *leg_fake_pt = new TLegend(0.2, 0.8, 0.47, 0.95);
  overlapHistograms(fCKF, fgreedySolver, fMLSolver, "fakerate", "pT", 0, 0.1, 0.0, 60.0,
                    leg_fake_pt, "plot/fakeRate_pt.pdf");
  TLegend *leg_fake_eta = new TLegend(0.65, 0.75, 0.92, 0.9);
  overlapHistograms(fCKF, fgreedySolver, fMLSolver, "fakerate", "eta", 0, 0.12, -3.2, 3.2,
                    leg_fake_eta, "plot/fakeRate_eta.pdf");
  TLegend *leg_fake_phi = new TLegend(0.65, 0.75, 0.92, 0.9);
  overlapHistograms(fCKF, fgreedySolver, fMLSolver, "fakerate", "phi", 0, 0.04, -3.2, 3.2,
                    leg_fake_phi, "plot/fakeRate_phi.pdf");

  // Plot the dupliacte rate
  TLegend *leg_dup_pt = new TLegend(0.65, 0.25, 0.92, 0.4);
  overlapHistograms(fCKF, fgreedySolver, fMLSolver, "duplicationRate", "pT",
                    0.00001, 2, 0.0, 60.0, leg_dup_pt, "plot/duplicateRate_pt.pdf");
  TLegend *leg_dup_eta = new TLegend(0.65, 0.70, 0.92, 0.85);
  overlapHistograms(fCKF, fgreedySolver, fMLSolver, "duplicationRate", "eta",
                    0.00001, 2, -3.2, 3.2, leg_dup_eta, "plot/duplicateRate_eta.pdf");
  TLegend *leg_dup_phi = new TLegend(0.65, 0.65, 0.92, 0.80);
  overlapHistograms(fCKF, fgreedySolver, fMLSolver, "duplicationRate", "phi",
                    0.00001, 2, -3.2, 3.2, leg_dup_phi, "plot/duplicateRate_phi.pdf");
}
