#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <iomanip>
#include <iostream>
#include <TH1D.h>
#include <string.h>
#include <sstream>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include "math.h"

#include "AtlasStyle.C"

using namespace std;

//---------------------------------------------------------
// Initialisation des constantes :

string name[5] = {"370259","370295","370323"};
string histo[6]={"mettst_pt","mt","Meff","m_pt","4_pt","n_tau"}  ;


TH1F*  Gbb_base[4][6];

TH1F*  tt_base[6];

int bin;

//---------------------------------------------------------
// Fonction qui va chercher un histogramme dans un fichier pour une remplir un autre :

void Rescale(TH1F* vec)

{ 
  double m1=0;
  m1=vec->Integral();
  vec->Scale(1/m1);
}


void get(TH1F* vec[] ,int j, string name ,string histo ,TFile* f )

{
  vec[j] = ((TH1F*)f->Get((name+histo).c_str())) ;


  Rescale(vec[j]);
}



//---------------------------------------------------------

int main ()
{
  int i=0;
  int j=0;
  int k=0;
  gROOT->LoadMacro("AtlasLabels.C");
  SetAtlasStyle();

  //--------------------------------------------------------- 

 TFile* Mtt = new TFile("../merged/ttbar/ttbar.root","READ");  

 for(j=0;j<6;j++)
   get( tt_base ,j , "" , histo[j] , Mtt ) ;
 

 for(k=0;k<3;k++)
   {
     TFile* MGbb = new TFile((("../merged/Gbb/base_")+name[k]+(".root")).c_str(),"READ");      
     for(j=0;j<6;j++)
       get( Gbb_base[k] ,j , "base_" , histo[j] , MGbb ) ;   
   }

 //  Rescale(Gbb_base,tt_base);


 TCanvas *c1 = new TCanvas("c1","analyse") ;
 TCanvas *c2 = new TCanvas("c2","analyse") ;
 TCanvas *c3 = new TCanvas("c3","analyse") ;
 TCanvas *c4 = new TCanvas("c4","analyse") ;
 TCanvas *c5 = new TCanvas("c5","analyse") ;
 TCanvas *c6 = new TCanvas("c6","analyse") ;

 for(j=0;j<6;j++)
   {	
  
     tt_base[j]->GetYaxis()->SetTitle("base");
     tt_base[j]->GetYaxis()->CenterTitle(true);	      
     tt_base[j]->GetYaxis()->SetTitleOffset(1.6); 
     tt_base[j]->SetFillColor(kBlue);
     tt_base[j]->SetLineColor(kBlue);

     Gbb_base[0][j]->SetLineColor(kRed);
     Gbb_base[1][j]->SetLineColor(kGreen);
     Gbb_base[2][j]->SetLineColor(kYellow);
   }

 TLegend *legend = new TLegend(0.70,0.60,1,0.90);
 legend->SetFillStyle(0);
 legend->SetBorderSize(0);
 legend->AddEntry(tt_base[0],"MC_tt");
 legend->AddEntry(Gbb_base[0][0],"m_g=1000GeV m_LSP=600GeV");
 legend->AddEntry(Gbb_base[1][0],"m_g=1500GeV m_LSP=600GeV");
 legend->AddEntry(Gbb_base[2][0],"m_g=1800GeV m_LSP=600GeV");
 legend->SetFillStyle(0);
 legend->SetBorderSize(0);


 c1->cd();
 tt_base[0]->Draw("HIST");
 Gbb_base[0][0]->Draw("SAME PE");
 Gbb_base[1][0]->Draw("SAME PE");
 Gbb_base[2][0]->Draw("SAME PE");
 legend->Draw("SAME");
 c1->Print("variable.pdf(");

 c2->cd();
 tt_base[1]->Draw("HIST");
 Gbb_base[0][1]->Draw("SAME PE");
 Gbb_base[1][1]->Draw("SAME PE");
 Gbb_base[2][1]->Draw("SAME PE");
 legend->Draw("SAME");
 c2->Print("variable.pdf");

 c3->cd();
 tt_base[2]->Draw("HIST");
 Gbb_base[0][2]->Draw("SAME PE");
 Gbb_base[1][2]->Draw("SAME PE");
 Gbb_base[2][2]->Draw("SAME PE");
 legend->Draw("SAME");
 c3->Print("variable.pdf");

 c4->cd();
 tt_base[3]->Draw("HIST");
 Gbb_base[0][3]->Draw("SAME PE");
 Gbb_base[1][3]->Draw("SAME PE");
 Gbb_base[2][3]->Draw("SAME PE");
 legend->Draw("SAME");
 c4->Print("variable.pdf");

 c5->cd();
 tt_base[4]->Draw("HIST");
 Gbb_base[0][4]->Draw("SAME PE");
 Gbb_base[1][4]->Draw("SAME PE");
 Gbb_base[2][4]->Draw("SAME PE");
 legend->Draw("SAME");
 c5->Print("variable.pdf");

 c6->cd();
 c6->SetLogy();
 tt_base[5]->Draw("HIST");
 tt_base[5]->GetYaxis()->SetRangeUser(0.001,1.1);  
 Gbb_base[0][5]->Draw("SAME PE");
 Gbb_base[1][5]->Draw("SAME PE");
 Gbb_base[2][5]->Draw("SAME PE");
 legend->Draw("SAME");
 c6->Print("variable.pdf)");

 c1->Destructor();
 c2->Destructor();
 c3->Destructor();
 c4->Destructor();
 c5->Destructor();
 c6->Destructor();
        
  return 0;
}


