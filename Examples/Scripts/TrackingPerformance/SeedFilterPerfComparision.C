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


void overlapHistograms(TFile *fSeed, TFile *fMLFilter, std::string histogramName, std::string variableName, float rangeMin, float rangeMax, float xMin, float xMax, TLegend* leg, std::string saveName){

    // Plot the histograme vs variableName
    std::string title = histogramName + "_vs_" + variableName;

    TCanvas *C = new TCanvas(title.c_str(), title.c_str());

    TEfficiency * hSeed = new TEfficiency(("Seed"+title).c_str(),("Seed"+title).c_str() , 100, 0, 100, 100, 0, 1);
    hSeed = (TEfficiency*)fSeed->Get((title).c_str());
    hSeed->SetLineColor(kBlack);
    hSeed->SetMarkerColor(kBlack);
    hSeed->SetMarkerStyle(20);
    hSeed->SetMarkerSize(0.8);
    hSeed->Draw("");
    setRange(hSeed, rangeMin, rangeMax, xMin, xMax);

    TEfficiency *hMLFilter = new TEfficiency(("MLSeedFilter" + title).c_str(),
                                             ("MLSeedFilter" + title).c_str(),
                                             100, 0, 100, 100, 0, 1);
    hMLFilter = (TEfficiency*)fMLFilter->Get((title).c_str());
    hMLFilter->SetLineColor(kAzure+5);
    hMLFilter->SetMarkerColor(kAzure+5);
    hMLFilter->SetMarkerStyle(22);
    hMLFilter->SetMarkerSize(0.8);
    hMLFilter->Draw("SAME");

    if(histogramName == "duplicationRate"){
        gPad->SetLogy();
        gPad->Update(); 
    }

    leg->AddEntry(hSeed,"CKF","lep");
    leg->AddEntry(hMLFilter, "MLSeedFilter", "lep");
    leg->Draw();

    C->SaveAs((saveName).c_str());

}

/// @brief Root macro to compare the performance of the different solvers
/// @param seed path to the seed output file
/// @param MLSolver path to the MLFilter output file
void SeedFilterPerfComparision(std::string seed = "", std::string MLFilter = ""){

    // Set the ATLAS style
  gROOT->LoadMacro("AtlasLabels.C");
  SetAtlasStyle();
  gStyle->SetErrorX(0);

  // Open the files
  TFile *fSeed = new TFile(seed.c_str());
  TFile *fMLFilter = new TFile(MLFilter.c_str());

  // Plot the seed efficiency
  TLegend *leg_eff_pt = new TLegend(0.65, 0.20, 0.92, 0.35);
  overlapHistograms(fSeed, fMLFilter, "trackeff", "pT", 0.4, 1.05, 0.0, 60.0,
                    leg_eff_pt, "plot/seedEfficiency_pt.pdf");
  TLegend *leg_eff_eta = new TLegend(0.65, 0.25, 0.92, 0.40);
  overlapHistograms(fSeed, fMLFilter, "trackeff", "eta", 0.4, 1.05, -3.2, 3.2,
                    leg_eff_eta, "plot/seedEfficiency_eta.pdf");
  TLegend *leg_eff_phi = new TLegend(0.65, 0.25, 0.92, 0.40);
  overlapHistograms(fSeed, fMLFilter, "trackeff", "phi", 0.8, 1.05, -3.2, 3.2,
                    leg_eff_phi, "plot/seedEfficiency_phi.pdf");
  TLegend *leg_eff_DeltaR = new TLegend(0.65, 0.25, 0.92, 0.40);
  overlapHistograms(fSeed, fMLFilter, "trackeff", "DeltaR", 0.8, 1.05, 0.0, 0.3,
                    leg_eff_DeltaR, "plot/seedEfficiency_deltaR.pdf");

}
