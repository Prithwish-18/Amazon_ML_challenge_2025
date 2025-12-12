import { BigBang } from './scenes/bigBang.js';
import { InflationMultiverse } from './scenes/inflationMultiverse.js';
import { LargeScale } from './scenes/largeScale.js';
import { GalaxyFormation } from './scenes/galaxyFormation.js';
import { StellarEvolution } from './scenes/stellarEvolution.js';
import { MilkyWay } from './scenes/milkyWay.js';
import { SolarBirth } from './scenes/solarBirth.js';
import { SolarOverview } from './scenes/solarOverview.js';
import { Earth } from './scenes/earth.js';

export class SceneManager {
  constructor({ renderer, scene, camera, pmrem, ui }) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.pmrem = pmrem;
    this.ui = ui;
    this.index = 0;
    this.timeline = [
      new BigBang(),
      new InflationMultiverse(),
      new LargeScale(),
      new GalaxyFormation(),
      new StellarEvolution(),
      new MilkyWay(),
      new SolarBirth(),
      new SolarOverview(),
      new Earth()
    ];
    ui.buildDots(this.timeline.map(s => s.title));
  }

  async start(){
    await this.timeline[0].enter(this.scene, this.camera, this.renderer, this.pmrem, this.ui);
    this.ui.activateDot(0);
  }

  caption(){ return this.timeline[this.index].caption; }

  async goTo(i){
    if(i<0 || i>=this.timeline.length || i===this.index) return;
    await this.timeline[this.index].exit(this.scene);
    this.index = i;
    this.ui.activateDot(this.index);
    await this.timeline[this.index].enter(this.scene, this.camera, this.renderer, this.pmrem, this.ui);
    this.ui.setCaption(this.caption());
  }

  async next(){ await this.goTo(Math.min(this.timeline.length-1, this.index+1)); }
  async prev(){ await this.goTo(Math.max(0, this.index-1)); }

  async skip(){ if (this.timeline[this.index].skip) await this.timeline[this.index].skip(); }

  update(dt){ const s = this.timeline[this.index]; s.update && s.update(dt); }

  resize(w,h){ this.timeline[this.index].resize && this.timeline[this.index].resize(w,h); }
}
